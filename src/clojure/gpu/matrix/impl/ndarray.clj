; N-dimensional array, with dimension checking in a lot of places
; not ideal for high performance computing where dimensions are defined

(ns gpu.matrix
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]]
            [clojure.string :as str]
            [gpu.matrix.impl.common :as common :refer [with-coerce-param]])
  (:import gpu.matrix.NDArray gpu.matrix.Initializer))

(set! *warn-on-reflection* true)

(common/extend-common-protocols gpu.matrix.NDArray)

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

(defn- all?
  "returns true if every element in coll (or the mapped version, if 
  a map function is provided) is true. Truthyness is tested using true?, so
  only the Bolean true will be tested as positive."
  ([coll]
   (every? true? coll))
  ([f coll]
   (every? true? (mapv f coll))))

(defn- make-index-type-hints [syms]
  "Make index type hints.
  eg: Converts the list
  
  '(a b c d)
  
  into
  
  '[a (long a) b (long b) c (long c)]
  
  which can be used together with other bindings (or even on its own)
  in macros"
  (vec
    (mapcat (fn [sym]
              (if (symbol? sym)
                `[~sym (long ~sym)]
                `[]))
            syms)))

(defn- in-bound?
  "Returns true if the matrix is in bound, i.e:
  0 <= idx < len" 
  [^long idx ^long len]
  (and (>= idx 0) (< idx len)))

(defmacro with-check-indices
  "
  Check the indices (third or more arguments) based on the provided
  matrix (second argument). Throws an error if it is out of bound.

  The way the third and more arguments is interpreted depends on the
  symbol ndims. If it is a number (note, not a symbol that points to
  a number) and is equals to 1 , then the third argument will be
  the row number. If it is equals to 2, then third argument will
  be the row number and fourth argument will be the col number.
  Otherwise, the third number is treated as an array of indices.
  
  In all cases, the other arguments will be treated as the body. 

  This macro also does some autocasting for certain symbols (only for 
  ndims = 1 or 2). For eg, if the row / col number is a symbol rather
  than a number, then the symbol will be casted to a long explicitly,
  preventing the usage of the reflection API where possible.

  usage:
  (with-check-indices 1 m row
    (println (.get m row)))

  (with-check-indices 2 m row col
    (println (.get m row)))

  ; When the first argument (ndims) is not 1 or 2 
  (with-check-indices (inc 0) m [row]
    (println (.get m row)))
  
  (with-check-ndims 3 m [i j k]
    (println (.get m i j k))) 
  " 
  [ndims m & args]
   (let [row-sym (gensym "row_")
         col-sym (gensym "col_")
         indexes-sym (gensym "indexes_")
         check-clause
         (cond (= ndims 1)
               `(and (>= ~row-sym 0)
                     (< ~row-sym (first (mp/get-shape ~m))))
               (= ndims 2)
               `(let [shape# (mp/get-shape ~m)]
                 (and (>= ~row-sym 0)
                      (< ~row-sym (first shape#))
                      (>= ~col-sym 0)
                      (< ~col-sym (second shape#))))
               :else
               `(let [shape# (mp/get-shape ~m)] 
                  (all? (map-indexed
                          (fn [dim# idx#]
                            (and (>= idx# 0)
                                 (< idx# (nth shape# dim#))))
                          ~indexes-sym))))
         body
         (cond (= ndims 2)
               (next (next args)) 
               :else
               (next args))
         all-indexes
         (cond (= ndims 1)
               `[~(first args)]
               (= ndims 2)
               `[~(first args) ~(second args)]
               :else
               (first args))
         type-hints
         (make-index-type-hints
           (cond (= ndims 1)
                 (take 1 args)
                 (= ndims 2)
                 (take 2 args)
                 :else
                 []))
         bindings
         (-> (cond (= ndims 1)
             `[~row-sym ~(first args)]
             (= ndims 2)
             `[~row-sym ~(first args)
               ~col-sym ~(second args)]
             :else
             `[~indexes-sym ~(first args)])
             (concat type-hints)
             (vec))]
    `(let ~bindings
       (if ~check-clause
         (do ~@body)
         (error
           (str
             "Index Error occured while trying to access the following:\n" 
             (->> (reduce (fn [memo# [dim# dim-count# idx#]]
                            (if (in-bound? idx# dim-count#)
                              memo#
                              (conj memo# [dim# dim-count# idx#])))
                           []
                           (map vector
                                (range (mp/dimensionality ~m))
                                (mp/get-shape ~m)
                                ~all-indexes))
                  (map (fn [[dim# dim-count# idx#]]
                         (str "-- index " idx# " of dimension " dim#
                              " , where the dimension-count of the dimensions is only " dim-count#)))
                  (str/join "\n"))
             "\n"))))))

(defmacro with-check-ndims [ndims m & body]
  "macro to check the dimensionality of the m against ndims.
  If it is equal, the body will be executed. Otherwise, a runtime error
  will be thrown."
  `(if (= ~ndims (.dimensionality ~m))
     (do ~@body)
     (error
       (str "Invalid dimension! Expecting dimension " (.dimensionality ~m) " "
            "but got " ~ndims " instead"))))

(defmacro with-check-shape [ndims [m-sym m] & body]
  "Binds m-sym to m, with a type hinting indicating NDArray. Executes body
  after running with-check-ndims and with-check-indices. Note that
  body will have to adhere to the format expected by with-check-indices.
  See (source with-check-indices) for more details
  
  Usage:
  
  (with-check-shape 2 [m (mp/construct-matrix (NDArray.) [ [1 2 3]])] row col
    (println (.get m row col)))
  
  Note that explicit type hinting is not required if row and col are symbols,
  this macro will automatically cast 'row and 'col to long types"
  `(let [~(with-meta m-sym {:tag NDArray}) ~m]
     (with-check-ndims ~ndims ~m-sym
       (with-check-indices ~ndims ~m-sym
         ~@body))))

(extend-protocol mp/PIndexedAccess
  NDArray
  (get-1d [m row]
    (with-check-shape
      1 [m m] row
      (.get m row)))
  (get-2d [m row col]
    (with-check-shape
      2 [m m] row col
      (.get m row col)))
  (get-nd [m indexes]
    (with-check-shape
      (count indexes) [m m] indexes
      (.get m (long-array indexes)))))

(extend-protocol mp/PIndexedSetting
  NDArray
  (set-1d [m row ^double v]
    (with-check-shape
      1 [m m] row
      (.set (.clone m) row v)))
  (set-2d [m row col ^double v]
    (with-check-shape
      2 [m m] row col
      (.set (.clone m) row col v)))
  (set-nd [m indexes ^double v]
    (with-check-shape
      (count indexes) [m m] indexes
      (.set (.clone m) (long-array indexes) v)))
  (is-mutable? [m] true))

(extend-protocol mp/PIndexedSettingMutable
  NDArray
  (set-1d! [m row ^double v]
    (with-check-shape
      1 [m m] row
      (.set m row v)))
  (set-2d! [m row col ^double v]
    (with-check-shape
      2 [m m] row col
      (.set m row col v)))
  (set-nd! [m indexes ^double v]
    (with-check-shape
      (count indexes) [m m] indexes
      (.set m (long-array indexes) v))))

(extend-protocol mp/PMatrixAdd
  NDArray
  (matrix-add [^NDArray m a]
    (with-coerce-param [a a] (.add m a)))
  (matrix-sub [^NDArray m a]
    (with-coerce-param [a a] (.sub m a))))

(extend-protocol mp/PMatrixMultiply
  NDArray
  (element-multiply [^NDArray m a]
    (with-coerce-param [a a] (.mul m a)))
  (matrix-multiply [^NDArray m a]
    (with-coerce-param [a a]
      (if (and (= (mp/dimensionality m) 2)
               (= (mp/dimensionality a) 2)
               (= (second (mp/get-shape m))
                  (first (mp/get-shape a))))
        (.mmul m a)
        (error "Incorrect dimensionality / shape in matrix-multiply")))))

(extend-protocol mp/PMatrixDivide
  NDArray
  (element-divide [^NDArray m a]
    (with-coerce-param [a a] (.div m a))))

(extend-protocol mp/PValueEquality
  NDArray
  (value-equals [^NDArray m a]
    (or (= m a)
        (with-coerce-param [a a]
          (.equals m a)))))

(extend-protocol mp/PMatrixEquality
  NDArray
  (matrix-equals [m a]
    (or (= m a)
        (with-coerce-param [a a]
          (.equals m a))))
  (matrix-equals-epsilon [^NDArray m ^NDArray a epsilon]
    (or (= m a)
        (with-coerce-param [a a]
          (.equals m a (double epsilon))))))

(extend-protocol mp/PDoubleArrayOutput
  NDArray
  (to-double-array [^NDArray m] (.flatten m))
  (as-double-array [^NDArray m] (.flatten m)))

(extend-protocol mp/PZeroDimensionAccess
  NDArray
  (get-0d [^NDArray m]
    (with-check-ndims 0 m
      (.get m 0)))
  (set-0d! [^NDArray m v]
    (with-check-ndims 0 m
      (.set m 0 (double v)))))

(extend-protocol mp/PMutableFill
  NDArray
  (fill! [^NDArray m v]
    (.fill m (double v))))

(extend-protocol mp/PImmutableAssignment
  NDArray
  (assign [^NDArray m src]
    (cond (= 0 (mp/dimensionality src))
          (.assign (.clone m) (double (mp/get-0d src)))  

          (= (mp/dimensionality src) (.dimensionality m))
          (if (mp/value-equals (mp/get-shape src) (.shape m))
            (with-coerce-param [src src]
              (.assign (.clone m) src))
            (error "dimension mismatch"))

          (< (mp/dimensionality src) (.dimensionality m))
          (if (mp/value-equals (drop (- (.dimensionality m) (mp/dimensionality src))
                                     (.shape m))
                               (mp/get-shape src))
            (with-coerce-param [src src]
              (.assign (.clone m) src))
            (error "dimension mismatch"))

          (> (mp/dimensionality src) (.dimensionality m))
          (error "Cannot assign a matrix of higher dimension to another matrix "
                 "of lower dimension"))))

