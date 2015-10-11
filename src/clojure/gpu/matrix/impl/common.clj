(ns gpu.matrix.impl.common
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
    (condp = ndims
      1 (Vector/newInstance (double-array data))
      (NDArray/newInstance
        (double-array (flatten data))
        ndims
        (long-array shape)))))

; assume it is an arbitary implementation
; construct a persistent vector representation of the data
(defmethod construct-matrix :default
  [data]
  (let [vm (mp/construct-matrix [] data)]
    (construct-matrix vm)))

(defmacro extend-common-protocols [klass]
  `(do
    (extend-protocol mp/PImplementation
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
        true))
     (extend-protocol mp/PCoercion
       ~klass
       (coerce-param [m# param#]
         (cond (or (instance? NDArray param#)
                   (instance? Vector param#))
               param#
               (number? param#) (double param#)
               :else
               (mp/construct-matrix m# param#))))
      (extend-protocol mp/PNumerical
        ~klass
        (numerical? [m#] true))
      (extend-protocol mp/PTypeInfo
        ~klass
        (element-type [m#] Double/TYPE))))

(defmacro with-coerce-param [bindings & body]
  (assert (= (count bindings) 2))
  (assert (symbol? (first bindings)))
  (let [param-sym (first bindings)
        param-val (second bindings)]
    `(cond (number? ~param-val)
           (let [~param-sym (double ~param-val)]
             ~@body)
           (instance? NDArray ~param-val)
           (let [~(vary-meta param-sym assoc :tag `NDArray) ~param-val]
             ~@body)
           (instance? Vector ~param-val)
           (let [~(vary-meta param-sym assoc :tag `Vector) ~param-val]
             ~@body)
           ; unkown type, we need to coerce it to NDArray
           :else
           (let [~(vary-meta param-sym assoc :tag `NDArray)
                 (mp/coerce-param (gpu.matrix.NDArray/sample) ~param-val)]
             ~@body))))

