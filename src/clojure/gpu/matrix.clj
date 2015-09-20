; This is the main interface to exntending core.matrix's protocols

(ns gpu.matrix
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m])
  (:import gpu.matrix.NDArray))

(defmulti construct-matrix class)
(defmethod construct-matrix NDArray
  [data]
  (.clone data))
(defmethod construct-matrix nil
  [data]
  nil)
(defmethod construct-matrix Number
  [data]
  (double data))
(defmethod construct-matrix :default
  [data]
  (let [ndims (m/dimensionality data)
          shape (m/shape data)]
      (NDArray/newInstance
        (double-array (flatten data))
        ndims
        (long-array shape))))

(extend-protocol mp/PImplementation
  NDArray
  (implementation-key [m] :gpu-matrix)
  (meta-info [m] {})
  (construct-matrix [m data]
    (construct-matrix data))
  (new-vector [m length]
    (NDArray/createFromShape (long-array 1 length)))
  (new-matrix [m rows columns]
    (NDArray/createFromShape (long-array 2 [rows columns])))
  (new-matrix-nd [m shape]
    (NDArray/createFromShape (long-array shape)))
  (supports-dimensionality? [m dimensions]
    true))
