package gpu.matrix;
import java.lang.reflect.Array;
import java.util.ArrayList;

public class ArrayHelper {
    public static long dimensionality(Object obj) {
        if (obj.getClass().isArray() && Array.getLength(obj) > 0) {
            return 1 + dimensionality(Array.get(obj, 0));
        } else {
            return 0;
        }
    }

    public static long[] shape(Object obj) {
        Object arr = obj;
        int ndims = (int) dimensionality(arr);
        long[] shape = new long[ndims];

        for(int i = 0 ; i < ndims-1 ; i++) {
            shape[i] = Array.getLength(arr);
            arr = Array.get(arr, 0);
        }
        shape[ndims-1] = Array.getLength(arr);

        return shape;
    }

    private static void flatten_recur(Object arr, ArrayList<Double> res) {
        if (!arr.getClass().isArray()) {
            res.add((double) arr);
        } else {
            long length = Array.getLength(arr);
            for(int i = 0 ; i < length; i++) {
                flatten_recur(Array.get(arr, i), res);
            }
        }
    }

    public static double[] flatten(Object arr) {
       final ArrayList<Double> res = new ArrayList<>();
       flatten_recur(arr, res);
       final double[] ret = new double[res.size()];

       for(int i = 0 ; i < res.size() ; i++) {
           ret[i] = res.get(i).doubleValue();
       }

       return ret;
    }

    public static long[] makeBasicStrides(long [] shape) {
        int length = shape.length;
        long[] res = new long[length];

        res[length-1] = 1;
        for (int i = length - 2 ; i >= 0 ; i--) {
            res[i] = res[i+1] * shape[i+1];
        }

        return res;
    }
}
