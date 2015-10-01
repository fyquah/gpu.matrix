package gpu.matrix;
import gpu.matrix.KernelLoader;

import gpu.matrix.KernelLoader;

public class JVMLoader {
    static {
        KernelLoader.loadLibrary("gpu-matrix");
        init();
    }

    // loads the kernels to load stuff JVM Resources
    // TODO: Merge this with KernelLoader, and rename KernelLoader
    // into something more appropriate (like JVMLoader?)
    public native static void init();
}
