package gpu.matrix;
import java.nio.ByteBuffer;
import gpu.matrix.Initializer;
import gpu.matrix.LoaderUtils;

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

    // Arimethic ops (returns new objects!), i.e: These methods are immutable
    public native Vector add(Vector a);
    public native Vector add(double a);
    public native Vector sub(Vector a);
    public native Vector sub(double a);
    public native Vector mul(Vector a);
    public native Vector mul(double a);
    public native Vector div(Vector a);
    public native Vector div(double a);

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
