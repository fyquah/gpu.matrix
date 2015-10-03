package gpu.matrix;

public interface Vector {
    public native Vector axpy(double a, Vector y);
    public native Vector scal(double a);
    public native Vector clone();
    public native double dot(Vector y);
    public native double nrm2();
    public native double asum();
}
