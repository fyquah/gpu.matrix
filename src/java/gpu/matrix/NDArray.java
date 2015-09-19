package gpu.matrix;
import java.nio.ByteBuffer;
import gpu.matrix.ArrayHelper;

public class NDArray {
    // Static methods
    static {
        System.loadLibrary("gpu-matrix-ndarray");
        init();
    }

    public native static NDArray sample();
    public static NDArray newInstance(Object arr) {
        final long[] shape = ArrayHelper.shape(arr);
        final long[] strides = ArrayHelper.makeBasicStrides(shape);

        return newInstance(
            ArrayHelper.flatten(arr),
            ArrayHelper.dimensionality(arr),
            shape,
            strides
        );
    }
    private native static NDArray newInstance(double[] data, long ndims, long[] shape, long[] strides);
    private native static void init();

    // constructors
    public NDArray(){
    }

    // instance
    // bb is the ByteBuffer used to store the pointer to the ndarray object in native code
    private ByteBuffer bb;

    public native NDArray add(NDArray y);
    public native void print();

    // for testing purposes
    public static void main(String[] args) {
        newInstance(new double[][]{ {1.0, 2.0, 3.0}, { 4.0, 5.0, 6.0}}).print();
    }
}
