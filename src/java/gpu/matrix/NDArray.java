package gpu.matrix;
import java.nio.ByteBuffer;

public class NDArray {
    // Static methods
    static {
        System.loadLibrary("gpu-matrix-ndarray");
        init();
    }
    public native static NDArray sample();
    private native static void init();

    // instance
    private ByteBuffer bb;

    public native NDArray add(NDArray y);
    public native void print();
}
