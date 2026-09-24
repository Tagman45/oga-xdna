/**
 * ogad_client.h — Client C++ pour OGA Daemon (ogad_v2.py).
 *
 * Communication JSON ligne par ligne via stdin/stdout du sous-processus.
 *
 * Usage:
 *   OGADClient client;
 *   client.start("python", {"ogad_v2.py"});  // lance le daemon
 *   client.load("C:/models/tinyllama");
 *   auto result = client.generate("Hello", 50);
 *   printf("TPS: %.1f\n", result.tps);
 */
#pragma once
#include <windows.h>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <chrono>

// -----------------------------------------------------------------------
// JSON parser minimal (sans dependance)
// -----------------------------------------------------------------------
class JsonValue {
public:
    enum Type { NULL_VAL, BOOL, NUMBER, STRING, OBJECT, ARRAY };
    Type type = NULL_VAL;
    bool bval = false;
    double nval = 0;
    std::string sval;
    std::vector<std::pair<std::string, JsonValue>> members;
    std::vector<JsonValue> items;

    JsonValue get(const std::string& key) const {
        for (auto& m : members)
            if (m.first == key) return m.second;
        return JsonValue();
    }
    
    bool is_string() const { return type == STRING; }
    bool is_number() const { return type == NUMBER; }
    bool is_object() const { return type == OBJECT; }
    
    std::string as_string() const { return type == STRING ? sval : ""; }
    double as_number() const { return type == NUMBER ? nval : 0; }
    int as_int() const { return (int)as_number(); }
    
    static JsonValue from_string(const std::string& json) {
        const char* p = json.c_str();
        return parse_impl(p);
    }

private:
    static void skip_ws(const char*& p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    
    static std::string parse_str(const char*& p) {
        if (*p != '"') return "";
        p++;
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\') { p++; if (*p) { s += *p; p++; } }
            else { s += *p; p++; }
        }
        if (*p == '"') p++;
        return s;
    }
    
    static JsonValue parse_impl(const char*& p) {
        skip_ws(p);
        JsonValue v;
        if (!*p) return v;
        
        if (*p == '"') {
            v.type = STRING;
            v.sval = parse_str(p);
        }
        else if (*p == '{') {
            v.type = OBJECT;
            p++;
            while (*p && *p != '}') {
                skip_ws(p);
                if (*p == '"') {
                    std::string key = parse_str(p);
                    skip_ws(p);
                    if (*p == ':') p++;
                    skip_ws(p);
                    auto val = parse_impl(p);
                    v.members.push_back({key, val});
                }
                skip_ws(p);
                if (*p == ',') p++;
            }
            if (*p == '}') p++;
        }
        else if (*p == '[') {
            v.type = ARRAY;
            p++;
            while (*p && *p != ']') {
                skip_ws(p);
                v.items.push_back(parse_impl(p));
                skip_ws(p);
                if (*p == ',') p++;
            }
            if (*p == ']') p++;
        }
        else if (*p == 't' && strncmp(p, "true", 4) == 0) {
            v.type = BOOL; v.bval = true; p += 4;
        }
        else if (*p == 'f' && strncmp(p, "false", 5) == 0) {
            v.type = BOOL; v.bval = false; p += 5;
        }
        else if (*p == 'n' && strncmp(p, "null", 4) == 0) {
            p += 4;
        }
        else {
            // Number
            v.type = NUMBER;
            char* end = nullptr;
            v.nval = strtod(p, &end);
            if (end) p = end;
        }
        return v;
    }
};

// -----------------------------------------------------------------------
// Command result
// -----------------------------------------------------------------------
struct OGACommandResult {
    bool ok = false;
    std::string error;
    
    // generate response
    std::string text;
    int n_tokens = 0;
    double tps = 0;
    double infer_ms = 0;
    int prompt_tokens = 0;
    
    // load response
    std::string model_id;
    double load_ms = 0;
    
    static OGACommandResult from_json(const std::string& json) {
        OGACommandResult r;
        auto v = JsonValue::from_string(json);
        auto status = v.get("status");
        r.ok = status.is_string() && status.as_string() == "ok";
        if (!r.ok) {
            auto msg = v.get("message");
            r.error = msg.is_string() ? msg.as_string() : "unknown error";
            return r;
        }
        
        r.text = v.get("text").as_string();
        r.n_tokens = v.get("n_tokens").as_int();
        r.tps = v.get("tps").as_number();
        r.infer_ms = v.get("infer_ms").as_number();
        r.prompt_tokens = v.get("prompt_tokens").as_int();
        r.model_id = v.get("model_id").as_string();
        r.load_ms = v.get("load_ms").as_number();
        return r;
    }
};

// -----------------------------------------------------------------------
// JSON builder minimal
// -----------------------------------------------------------------------
class JsonBuilder {
    std::string buf;
    
    static std::string escape(const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '"' || c == '\\') { r += '\\'; r += c; }
            else if (c == '\n') r += "\\n";
            else if (c == '\r') r += "\\r";
            else if (c == '\t') r += "\\t";
            else r += c;
        }
        return r;
    }
    
public:
    JsonBuilder() { buf = "{"; }
    
    JsonBuilder& add(const std::string& key, const std::string& val) {
        if (buf.size() > 1) buf += ",";
        buf += "\"" + escape(key) + "\":\"" + escape(val) + "\"";
        return *this;
    }
    JsonBuilder& add(const std::string& key, int val) {
        if (buf.size() > 1) buf += ",";
        buf += "\"" + key + "\":" + std::to_string(val);
        return *this;
    }
    JsonBuilder& add(const std::string& key, double val) {
        if (buf.size() > 1) buf += ",";
        buf += "\"" + key + "\":" + std::to_string(val);
        return *this;
    }
    
    std::string build() { return buf + "}\n"; }
};

// -----------------------------------------------------------------------
// OGAD Client
// -----------------------------------------------------------------------
class OGADClient {
    HANDLE h_child_stdin_write = NULL;
    HANDLE h_child_stdout_read = NULL;
    PROCESS_INFORMATION pi = {};
    bool running = false;
    std::string model_path;

public:
    ~OGADClient() { stop(); }

    bool start(const std::string& python_exe = "python",
               const std::string& script = "C:\\trixdna_test\\ogad_v2.py") {
        if (running) return true;

        SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};

        HANDLE h_stdin_read = NULL, h_stdout_write = NULL;
        if (!CreatePipe(&h_stdin_read, &h_child_stdin_write, &sa, 0)) return false;
        if (!CreatePipe(&h_child_stdout_read, &h_stdout_write, &sa, 0)) {
            CloseHandle(h_stdin_read); CloseHandle(h_child_stdin_write); return false;
        }

        // Ensure child handles are inherited
        SetHandleInformation(h_child_stdin_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(h_child_stdout_read, HANDLE_FLAG_INHERIT, 0);

        std::string cmd = python_exe + " " + script;

        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.hStdInput = h_stdin_read;
        si.hStdOutput = h_stdout_write;
        si.hStdError = h_stdout_write;
        si.dwFlags = STARTF_USESTDHANDLES;

        BOOL ok = CreateProcessA(NULL, (char*)cmd.c_str(), NULL, NULL, TRUE,
                                 CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
        
        CloseHandle(h_stdin_read);
        CloseHandle(h_stdout_write);

        if (!ok) {
            CloseHandle(h_child_stdin_write);
            CloseHandle(h_child_stdout_read);
            return false;
        }

        running = true;
        return true;
    }

    void stop() {
        if (!running) return;
        
        // Send exit command
        send_raw("{\"cmd\":\"exit\"}\n");
        
        // Wait for process
        if (pi.hProcess) {
            WaitForSingleObject(pi.hProcess, 3000);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
        
        if (h_child_stdin_write) CloseHandle(h_child_stdin_write);
        if (h_child_stdout_read) CloseHandle(h_child_stdout_read);
        running = false;
    }

    OGACommandResult load(const std::string& path) {
        model_path = path;
        auto cmd = JsonBuilder().add("cmd", "load").add("model", path).build();
        return send_and_recv(cmd);
    }

    OGACommandResult generate(const std::string& prompt, int max_tokens = 50,
                               double temperature = 0.7) {
        auto cmd = JsonBuilder()
            .add("cmd", "generate")
            .add("model", model_path)
            .add("prompt", prompt)
            .add("max_tokens", max_tokens)
            .add("temperature", temperature)
            .build();
        return send_and_recv(cmd);
    }

    OGACommandResult ping() {
        return send_and_recv("{\"cmd\":\"ping\"}\n");
    }

private:
    void send_raw(const std::string& data) {
        DWORD written = 0;
        WriteFile(h_child_stdin_write, data.data(), (DWORD)data.size(), &written, NULL);
    }

    std::string read_line() {
        std::string line;
        char c;
        DWORD read = 0;
        while (true) {
            if (!ReadFile(h_child_stdout_read, &c, 1, &read, NULL) || read == 0) break;
            if (c == '\n') break;
            line += c;
        }
        return line;
    }

    OGACommandResult send_and_recv(const std::string& cmd) {
        send_raw(cmd);
        auto resp = read_line();
        return OGACommandResult::from_json(resp);
    }
};

// -----------------------------------------------------------------------
// Test / demo
// -----------------------------------------------------------------------
#ifdef OGAD_CLIENT_TEST
int main() {
    printf("=== OGAD Client Test ===\n\n");
    
    OGADClient client;
    
    printf("[1] Starting daemon...\n");
    if (!client.start()) {
        printf("  FAILED\n");
        return 1;
    }
    printf("  OK\n\n");
    
    printf("[2] Ping...\n");
    auto p = client.ping();
    printf("  Status: %s\n", p.ok ? "OK" : p.error.c_str());
    
    if (!p.ok) return 1;
    
    printf("\n[3] Loading model...\n");
    auto r = client.load("C:\\trixdna_test\\tinyllama_oga");
    printf("  Status: %s\n", r.ok ? "OK" : r.error.c_str());
    if (r.ok) printf("  Load: %.0fms\n", r.load_ms);
    
    if (!r.ok) { client.stop(); return 1; }
    
    printf("\n[4] Generating...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto g = client.generate("Hello, my name is", 30);
    auto dt = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    
    if (g.ok) {
        printf("  Text: %s\n", g.text.c_str());
        printf("  Tokens: %d\n", g.n_tokens);
        printf("  TPS: %.1f\n", g.tps);
        printf("  Infer: %.0fms\n", g.infer_ms);
        printf("  Round-trip: %.0fms\n", dt);
    } else {
        printf("  Error: %s\n", g.error.c_str());
    }
    
    printf("\n[5] Done.\n");
    client.stop();
    return 0;
}
#endif // OGAD_CLIENT_TEST
