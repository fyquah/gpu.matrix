package gpu.matrix;
import java.nio.ByteBuffer;
import gpu.matrix.ArrayHelper;
import gpu.matrix.JVMLoader;

public class NDArray {
    // Static methods
    static {
        System.loadLibrary("gpu-matrix");
        init();
        JVMLoader.init();
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

    public native static NDArray newScalar(double v);
    public native static NDArray newScalar();
    public native static NDArray createFromShape(long[] shape);
    public native static NDArray newInstance(double[] data, long ndims, long[] shape, long[] strides);
    private native static void init();

    // constructors
    public NDArray(){}

    // instance
    // bb is the ByteBuffer used to store the pointer to the ndarray object in native code
    private ByteBuffer bb;

    public native long dimensionality();
    public native long[] shape();
    public native NDArray clone();

    // getters and setters
    public native double get(long i);
    public native double get(long i, long j);
    public native double get(long [] indexes);
    public native NDArray set(long i, double v);
    public native NDArray set(long i, long j, double v);
    public native NDArray set(long[] shape, double v);

    // value equality
    public native boolean equals(NDArray y);
    public native boolean equals(double y);

    // returns a flatten representation in row-major order
    public native double[] flatten();

    // arimethic ops
    public native NDArray add(NDArray y);
    public native NDArray add(double y);
    public native NDArray sub(NDArray y);
    public native NDArray sub(double y);
    public native NDArray mul(NDArray y);
    public native NDArray mul(double y);
    public native NDArray div(NDArray y);
    public native NDArray div(double y);

    @Override
    protected native void finalize();

    // For debuggin purposes
    public native void print();
}
