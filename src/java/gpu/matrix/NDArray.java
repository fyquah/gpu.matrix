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
    public static NDArray newInstance(double[] data, long ndims, long[] shape) {
        return newInstance(
            data, ndims, shape, ArrayHelper.makeBasicStrides(shape)
        );
    }

    public native static NDArray createFromShape(long[] shape);
    public native static NDArray newInstance(double[] data, long ndims, long[] shape, long[] strides);
    private native static void init();

    // constructors
    public NDArray(){
    }

    // instance
    // bb is the ByteBuffer used to store the pointer to the ndarray object in native code
    private ByteBuffer bb;

    public native long dimensionality();
    public native long[] shape();
    public native NDArray clone();

    public native NDArray add(NDArray y);
    public native double get(long i);
    public native double get(long i, long j);
    public native double get(long [] indexes);
    public native NDArray set(long i, double v);
    public native NDArray set(long i, long j, double v);
    public native NDArray set(long[] shape, double v);

    // For debuggin purposes
    public native void print();
    @Override
    protected native void finalize();

    // for testing purposes
    public static void main(String[] args) {
        NDArray m = NDArray.newInstance(new double[][]{{ 0.7, 5, 6}, { 1, 2, 3}});
        m.print();
        m.set(0, 2, 855.0);
        m.set(new long[]{0, 1}, 490.23);
        m.print();
    }
}
