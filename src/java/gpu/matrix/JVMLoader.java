package gpu.matrix;

public class JVMLoader {
    static {
        System.loadLibrary("gpu-matrix");
    }

    public native static void init();
}
