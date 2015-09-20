; This is the main interface to exntending core.matrix's protocols

(ns gpu.matrix
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]])
  (:import gpu.matrix.NDArray))

(set! *warn-on-reflection* true)

(defmulti construct-matrix class)
(defmethod construct-matrix NDArray
  ^NDArray [^NDArray data]
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

(extend-protocol mp/PDimensionInfo
  NDArray
  (dimensionality [m] (.dimensionality m))
  (get-shape [m] (.shape m))
  (is-scalar? [m] (= 0 (.dimensionality m)))
  (is-vector? [m] (= 1 (.dimensionality m)))
  (dimension-count [m dimension-number]
    (let [shape (.shape m)
          ndims (.dimensionality m)]
      (if (or (< dimension-number 0)
              (>= dimension-number ndims)) 
        (error (str "Invalid dimension! Expecting dimension to be between inclusive 0 and "
                    (dec ndims) ", but got " dimension-number "instead!"))
        (aget shape dimension-number)))))

(extend-protocol mp/PIndexedAccess
  NDArray
  (get-1d [^NDArray m row]
    (let [ndims (.dimensionality m)
          row (long row)]
      (if (= ndims row)
        (.get m row) 
        (error "Invalid shape!"))))
  (get-2d [^NDArray m row col]
    (let [ndims (.dimensionality m)
          row (long row)
          col (long col)]
      (if (= ndims 2)
        (.get m row col) 
        (error "Invalid shape!"))))
  (get-nd [^NDArray m indexes]
    (let [ndims (.dimensionality m)
          ^"[J" indexes (long-array indexes)]
      (if (= ndims (count indexes))
        (.get m indexes)
        (error "Invalid shape!")))))

