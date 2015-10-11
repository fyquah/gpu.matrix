package gpu.matrix;
import java.nio.ByteBuffer;
import gpu.matrix.Initializer;
import gpu.matrix.LoaderUtils;
import gpu.matrix.NDArray;

public class Vector {
    static {
        LoaderUtils.loadLibrary("gpu-matrix");
        Initializer.init();
    }
    // BLAS procedures, note that these are mutable (except for clone)
    public native Vector axpy(double a, Vector y);
    public native Vector scal(double a);
    public native Vector clone();
    public native double dot(Vector y);
    public native double nrm2();
    public native double asum();

    // Arimethic ops (returns new objects!), i.e: These methods do
    // not mutate the origina object

    public native NDArray add(NDArray a);
    public native Vector add(Vector a);
    public native Vector add(double a);

    public native NDArray sub(NDArray a);
    public native Vector sub(Vector a);
    public native Vector sub(double a);

    public native NDArray mul(NDArray a);
    public native Vector mul(Vector a);
    public native Vector mul(double a);

    public native NDArray div(NDArray a);
    public native Vector div(Vector a);
    public native Vector div(double a);

    // element map ops, returns new objects, does not mutate present object
    public native Vector pow(double alpha);

    // reduce ops
    public native double sum();

    // various equality ops
    public native boolean valueEquality(double [] arr);
    public native boolean valueEquality(Vector v);
    public native boolean valueEquality(NDArray arr);
    public boolean valueEquality(Object arr) {
        return true;
    }

    // information about Vector
    public native long length();
    public native double get(long idx);
    public native void set(long idx, double value);

    // Alternate representation
    public native double[] toArray();

    // Debugging stuff
    public native Vector print();

    // constructors 
    public static native Vector newInstance(double[] data);
    public static native Vector newInstance(long length);
    public static native Vector newInstance(long length, double data);

    // fields
    private ByteBuffer bb;
    @Override
    protected native void finalize();
}
