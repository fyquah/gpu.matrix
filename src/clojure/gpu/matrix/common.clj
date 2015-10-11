(ns gpu.matrix.common
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]]
            [clojure.string :as str])
  (:import gpu.matrix.Vector gpu.matrix.NDArray))

(defmulti construct-matrix class)

(defmethod construct-matrix NDArray 
  ^NDArray [^NDArray data]
  (.clone data))

(defmethod construct-matrix Vector 
  ^Vector [^Vector data]
  (.clone data))

(defmethod construct-matrix nil
  [data]
  nil)
(defmethod construct-matrix Number
  [data]
  (double data))
(defmethod construct-matrix clojure.lang.PersistentVector
  [data]
  (let [data (m/to-nested-vectors data)
        ndims (m/dimensionality data)
        shape (m/shape data)]
      (NDArray/newInstance
        (double-array (flatten data))
        ndims
        (long-array shape))))

; assume it is an arbitary implementation
; construct a persistent vector representation of the data
(defmethod construct-matrix :default
  [data]
  (let [vm (mp/construct-matrix [] data)]
    (construct-matrix vm)))

(defmacro extend-common-protocols [klass]
  `(extend-protocol mp/PImplementation
    ~klass 
    (implementation-key [m#] :gpu-matrix)
    (meta-info [m#] {})
    (construct-matrix [m# data#]
      (construct-matrix data#))
    (new-vector [m# length#]
      (Vector/newInstance (long length#)))
    (new-matrix [m# rows# columns#]
      (NDArray/createFromShape (long-array 2 [rows# columns#])))
    (new-matrix-nd [m# shape#]
      (NDArray/createFromShape (long-array shape#)))
    (supports-dimensionality? [m# dimensions#]
      true)))

