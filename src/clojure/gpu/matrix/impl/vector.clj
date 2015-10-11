; Specialized Real Vector implementation
; based on the class gpu.matrix.Vector

(ns gpu.matrix.vector
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]]
            [clojure.string :as str]
            [gpu.matrix.impl.common :as common :refer [with-coerce-param]])
  (:import gpu.matrix.NDArray gpu.matrix.Initializer gpu.matrix.Vector))

(set! *warn-on-reflection* true)

(common/extend-common-protocols gpu.matrix.Vector)

(defmacro dimension-error [dimension-number]
  `(error (str "Invalid dimension! Expecting dimension to be 0 for gpu.matrix.Vector"
               ", but got " ~dimension-number "instead!")))

(defmacro in-bound? [m idx]
  `(let [idx# (long ~idx)]
    (and (>= idx# 0)
         (< idx# (.length ~m)))))

(defmacro with-check-index
  [m idx & body]
  `(let [~(vary-meta m assoc :tag `Vector) ~m
         ~idx (long ~idx)]
     (if (in-bound? ~m ~idx)
       ~@body
       (error (str "index " ~idx " is out of bound!")))))

(defmacro with-check-indices-list
  [m indexes & body]
  )

(extend-protocol mp/PDimensionInfo
  Vector 
  (dimensionality [m] 1)
  (get-shape [m] [(.length m)])
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [dimension-number (long dimension-number)]
      (if (= 0 dimension-number)
        (.length m)
        (dimension-error dimension-number)))))

(extend-protocol mp/PIndexedAccess
  Vector 
  (get-1d [m idx]
    (with-check-index m idx 
      (.get m idx)))
  (get-2d [m row col]
    (dimension-error 2))
  (get-nd [m indexes]
    (if (= 1 (count indexes))
      (let [idx (first indexes)]
        (with-check-index m idx 
          (.get m idx)))
      (error "get-nd for gpu.matrix.Vector expects only 1 element in indices array / vector / list."
             "Got " (count indexes) " instead."))))


(extend-protocol mp/PIndexedSettingMutable
  Vector
  (set-1d! [m idx ^double v]
    (with-check-index m idx
      (.set m idx v)))
  (set-2d! [m row col ^double v]
    (dimension-error 2))
  (set-nd! [m indexes ^double v]
    (if (= 1 (count indexes))
      (let [^long idx (first indexes)]
        (with-check-index m idx
          (.set m idx v)))
      (error "get-nd for gpu.matrix.Vector expects only 1 element in indices array / vector / list."
             "Got " (count indexes) " instead."))))

(extend-protocol mp/PIndexedSetting
  Vector
  (set-1d [m idx ^double v]
    (let [^Vector m (.clone m)]
      (with-check-index m idx
        (.set m idx v))))
  (set-2d [m row col ^double v]
    (dimension-error 2))
  (set-nd [^Vector m indexes ^double v]
    (if (= 1 (count indexes))
      (let [^Vector m (.clone m)
            ^long idx (first indexes)]
        (with-check-index m idx
          (.set m idx v)))
      (error "get-nd for gpu.matrix.Vector expects only 1 element in indices array / vector / list."
             "Got " (count indexes) " instead.")))
  (is-mutable? [m] true))

(extend-protocol mp/PMatrixCloning
  Vector
  (clone [^Vector m] (.clone m)))

(extend-protocol mp/PMatrixAdd
  Vector 
  (matrix-add [^Vector m a]
    (with-coerce-param [a a] (.add m a)))
  (matrix-sub [^Vector m a]
    (with-coerce-param [a a] (.sub m a))))

(extend-protocol mp/PDoubleArrayOutput
  Vector
  (to-double-array [^Vector m] (.toArray m))
  (to-double-array [^Vector m] (.toArray m)))

(extend-protocol mp/PNegation
  Vector
  (negate [^Vector m] (.mul m -1.0)))

(extend-protocol mp/PExponent
  Vector
  (element-pow [^Vector m a]
    (.pow m (double a))))

(extend-protocol mp/PSummable
  Vector
  (element-sum [^Vector m]
    (.sum m)))

(extend-protocol mp/PMatrixDivide
  Vector 
  (element-divide [^Vector m a]
    (with-coerce-param [a a] (.div m a))))

(extend-protocol mp/PMatrixMultiply
  Vector 
  (element-multiply [^Vector m a]
    (with-coerce-param [a a] (.mul m a))))

(extend-protocol mp/PValueEquality
  Vector
  (value-equals [^Vector m a]
    (or (= m a)
        (condp isa? (class a)
          gpu.matrix.NDArray
          (.valueEquality m ^NDArray a)

          gpu.matrix.Vector
          (.valueEquality m ^Vector a)

          (Class/forName "[D")
          (.valueEquality m ^"[D" a) 

          clojure.lang.PersistentVector
          (let [a ^clojure.lang.PersistentVector a]
            (and (= 1 (mp/dimensionality a))
                 (= (count a) (.length m))
                 (.valueEquality m ^"[D" (double-array a))))

          ; else
          (and (= 1 (mp/dimensionality a))
               (= (mp/dimension-count a 0) (.length m))
               (.valueEquality m ^"[D" (mp/to-double-array m)))))))

(extend-protocol mp/PMatrixProducts
  Vector
  (inner-product [^Vector m a]
    (condp isa? (class a)
      gpu.matrix.Vector
      (let [a ^Vector a]
        (if (= (.length m) (.length a))
          (.dot m a)
          (error "inner-product expects length of both argument vectors to be equal!")))

      gpu.matrix.NDArray
      (let [a ^NDArray a]
        (and (or (= (.dimensionality a) 1)
                 (error "Expecting a 1-dimensional NDArray as argument of inner product "
                        "but got a/an " (.dimensionality a) "-dim gpu.matrix.NDArray instead!"))
             (or (= (aget ^"[J" (.shape a) 0) (.length m))
                 (error "inner-product expects length of both argument vectors to be equal!"))
             (.dot m a)))

      (and (or (= (mp/dimensionality a) 1)
               (error "Expecting a 1-dimensional object as argument of inner product but got "
                      (mp/dimensionality a) " instead!"))
           (or (= (first (mp/get-shape a)) (.length m))
               (error "inner-product expects length of both argument vectors to be equal!"))
           (if-let [v (or (mp/as-double-array a)
                          (mp/to-double-array a))]
             (.dot m ^"[D" v)
             nil)))))

