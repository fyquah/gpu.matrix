package gpu.matrix;
import java.nio.ByteBuffer;
import gpu.matrix.Initializer;

public class Vector {
    static {
        System.loadLibrary("gpu-matrix");
        Initializer.init();
    }
    // BLAS procedures, note that these are mutable (except for clone)
    public native Vector axpy(double a, Vector y);
    public native Vector scal(double a);
    public native Vector clone();
    public native double dot(Vector y);
    public native double nrm2();
    public native double asum();

    // Arimethic ops

    // Debugging stuff
    public native void print();

    // constructors 
    public static native Vector newInstance(double[] data);

    // fields
    private ByteBuffer bb;
    @Override
    protected native void finalize();

    // main
    public static void main(String[] args) {
        Vector v_x = newInstance(new double[]{ 1,2,3,4 });
        Vector v_y = newInstance(new double[]{ 5,6,7,8 });
        v_x.print();
        System.out.println("v_x.scal(10)");
        v_x.scal(10.0);
        v_x.print();
        v_x.axpy(11.2, v_y);
        v_x.print();
        System.out.println("v_x dot v_y = " +  String.valueOf(v_x.dot(v_y)));
        System.out.println("Magnitude = " + String.valueOf(v_x.nrm2()));
        System.out.println("asum = " + String.valueOf(v_x.asum()));
    }
}
