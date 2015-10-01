package gpu.matrix;
import gpu.matrix.KernelLoader;

public class JVMLoader {
    static {
        KernelLoader.loadLibrary("gpu-matrix");
    }

    public native static void init();
}
