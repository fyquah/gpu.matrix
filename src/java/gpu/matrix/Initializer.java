package gpu.matrix;
import gpu.matrix.LoaderUtils;

public class Initializer {
    static {
        LoaderUtils.loadLibrary("gpu-matrix");
        init();
    }

    public native static void init();
}
