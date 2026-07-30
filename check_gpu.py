import sys
try:
    import llama_cpp
    from llama_cpp import llama_supports_gpu_offload, llama_backend_init
except ImportError:
    print("ERROR: llama-cpp-python is not installed.")
    sys.exit(1)

def check_gpu():
    print("--- llama-cpp-python GPU Check ---")
    
    # 1. Check general GPU offload support compiled into the wheel
    try:
        supports_gpu = llama_supports_gpu_offload()
        print(f"Compiled with GPU offload support: {supports_gpu}")
    except AttributeError:
        print("Compiled with GPU offload support: Unknown (function missing)")

    # 2. Try to initialize the backend and see what it reports
    print("\nAttempting to initialize llama.cpp backend...")
    print("If you see 'ggml_vulkan' in the logs below, Vulkan is working!\n")
    
    try:
        # Initializing the backend will print system info to stdout/stderr
        # including things like "ggml_vulkan: Found 1 devices"
        llama_backend_init()
    except Exception as e:
        print(f"Backend init failed: {e}")

if __name__ == "__main__":
    check_gpu()
