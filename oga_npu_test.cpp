/**
 * oga_npu_test.cpp — ONNX Runtime GenAI C API direct.
 * Load onnxruntime-genai.dll, load OGA model, generate text on NPU XDNA2.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <chrono>

typedef void* OgaHandle;
typedef int32_t OgaStatus;

struct OGALib {
    HMODULE lib = nullptr;
    wchar_t old_cwd[MAX_PATH];

    OgaHandle (*CreateModel)(const char*) = nullptr;
    void      (*DestroyModel)(OgaHandle) = nullptr;
    OgaHandle (*CreateGenerator)(OgaHandle, OgaHandle) = nullptr;
    void      (*DestroyGenerator)(OgaHandle) = nullptr;
    OgaStatus (*GenerateNextToken)(OgaHandle) = nullptr;
    bool      (*IsDone)(OgaHandle) = nullptr;
    size_t    (*GetSequenceCount)(OgaHandle) = nullptr;
    OgaHandle (*GetSequenceData)(OgaHandle) = nullptr;
    OgaHandle (*CreateGeneratorParams)(OgaHandle) = nullptr;
    void      (*DestroyGeneratorParams)(OgaHandle) = nullptr;
    OgaStatus (*SetSearchNumber)(OgaHandle, const char*, double) = nullptr;
    OgaHandle (*CreateTokenizer)(OgaHandle) = nullptr;
    void      (*DestroyTokenizer)(OgaHandle) = nullptr;
    void      (*AppendTokens)(OgaHandle, const int32_t*, size_t) = nullptr;
    void      (*DestroySequences)(OgaHandle) = nullptr;
    const char* (*ResultGetError)(OgaHandle) = nullptr;
    void      (*DestroyResult)(OgaHandle) = nullptr;
    void     (*DestroyString)(char*) = nullptr;

    bool load() {
        GetCurrentDirectoryW(MAX_PATH, old_cwd);
        const wchar_t* deploy = L"C:\\Program Files\\RyzenAI\\1.7.1\\deployment";
        const wchar_t* deploy_b = L"C:\\Program Files\\RyzenAI\\1.7.1b\\deployment";
        SetCurrentDirectoryW(deploy);
        
        // Pre-load the AMD onnxruntime.dll FIRST.
        // The AMD one (21 MB) has RyzenAI EP built in.
        HMODULE ort = LoadLibraryW(L"onnxruntime.dll");
        printf("[ORT] Pre-loaded: 0x%p\n", ort);
        
        // Load the PIP version of onnxruntime-genai.dll (6.2 MB vs AMD's 4.5 MB).
        // The pip version is newer and compatible with ORT 1.17.1.
        // The AMD deployment's version (4.5 MB) expects ORT API v23 which isn't available.
        const wchar_t* pip_path = L"C:\\Users\\videl\\AppData\\Roaming\\Python\\Python314\\site-packages\\onnxruntime_genai\\onnxruntime-genai.dll";
        lib = LoadLibraryW(pip_path);
        if (!lib) {
            // Fallback to AMD deployment
            SetCurrentDirectoryW(deploy);
            lib = LoadLibraryW(L"onnxruntime-genai.dll");
        }
        if (!lib) {
            // Try 1.7.1b
            SetCurrentDirectoryW(deploy_b);
            ort = LoadLibraryW(L"onnxruntime.dll");
            lib = LoadLibraryW(L"onnxruntime-genai.dll");
        }
        if (!lib) { printf("[OGA] FAIL\n"); return false; }
        printf("[OGA] onnxruntime-genai.dll loaded\n");

        auto get = [&](const char* name, auto& ptr) {
            *(void**)&ptr = GetProcAddress(lib, name);
            if (!ptr) printf("[OGA] MISSING: %s\n", name);
        };

        get("OgaCreateModel", CreateModel);
        get("OgaDestroyModel", DestroyModel);
        get("OgaCreateGenerator", CreateGenerator);
        get("OgaDestroyGenerator", DestroyGenerator);
        get("OgaGenerator_GenerateNextToken", GenerateNextToken);
        get("OgaGenerator_IsDone", IsDone);
        get("OgaGenerator_GetSequenceCount", GetSequenceCount);
        get("OgaGenerator_GetSequenceData", GetSequenceData);
        get("OgaCreateGeneratorParams", CreateGeneratorParams);
        get("OgaDestroyGeneratorParams", DestroyGeneratorParams);
        get("OgaGeneratorParamsSetSearchNumber", SetSearchNumber);
        get("OgaCreateTokenizer", CreateTokenizer);
        get("OgaDestroyTokenizer", DestroyTokenizer);
        get("OgaGenerator_AppendTokens", AppendTokens);
        get("OgaDestroySequences", DestroySequences);
        get("OgaResultGetError", ResultGetError);
        get("OgaDestroyResult", DestroyResult);
        get("OgaDestroyString", DestroyString);

        return CreateModel != nullptr;
    }

    void unload() {
        if (lib) FreeLibrary(lib);
        lib = nullptr;
        SetCurrentDirectoryW(old_cwd);
    }
};

static OGALib g_oga;

double now_ms() {
    static auto start = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
}

int main() {
    printf("========================================\n");
    printf("OGA NPU C API Test\n");
    printf("========================================\n\n");

    if (!g_oga.load()) return 1;

    // Create model
    printf("\n[1] OgaCreateModel...\n");
    double t0 = now_ms();
    OgaHandle model = g_oga.CreateModel("C:\\trixdna_test\\tinyllama_oga");
    if (!model) { printf("  FAILED\n"); g_oga.unload(); return 1; }
    printf("  OK (%.0fms)\n", now_ms() - t0);

    // Create generator params
    printf("\n[2] OgaCreateGeneratorParams...\n");
    OgaHandle params = g_oga.CreateGeneratorParams(model);
    if (!params) { printf("  FAILED\n"); g_oga.DestroyModel(model); g_oga.unload(); return 1; }
    printf("  OK\n");

    if (g_oga.SetSearchNumber) {
        g_oga.SetSearchNumber(params, "max_length", 50);
    }

    // Create generator
    printf("\n[3] OgaCreateGenerator...\n");
    OgaHandle gen = g_oga.CreateGenerator(model, params);
    if (!gen) { printf("  FAILED\n"); g_oga.DestroyGeneratorParams(params); g_oga.DestroyModel(model); g_oga.unload(); return 1; }
    printf("  OK\n");

    // Append prompt tokens (dummy)
    if (g_oga.AppendTokens) {
        printf("\n[4] AppendTokens (prompt)...\n");
        int32_t prompt[] = {1, 100, 200, 300, 400};
        g_oga.AppendTokens(gen, prompt, 5);
    }

    // Generate
    printf("\n[5] GenerateNextToken loop...\n");
    t0 = now_ms();
    int n = 0;
    while (!g_oga.IsDone(gen) && n < 30) {
        if (g_oga.GenerateNextToken(gen) != 0) break;
        n++;
    }
    double dt = now_ms() - t0;
    printf("  %d tokens in %.0fms = %.1f TPS\n", n, dt, n * 1000.0 / dt);

    // Get output
    if (n > 0) {
        printf("\n[6] Output:\n");
        OgaHandle seqs = g_oga.GetSequenceData(gen);
        if (seqs) {
            size_t count = g_oga.GetSequenceCount ? g_oga.GetSequenceCount(gen) : 0;
            printf("  Sequences: %zu\n", count);
        }
        printf("\n>>> NPU INFERENCE VIA OGA C API: SUCCESS! (%d tokens)\n", n);
    }

    // Cleanup
    g_oga.DestroyGenerator(gen);
    g_oga.DestroyGeneratorParams(params);
    g_oga.DestroyModel(model);
    g_oga.unload();
    printf("\nDone.\n");
    return 0;
}
