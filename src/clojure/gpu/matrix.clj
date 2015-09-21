; This is the main interface to exntending core.matrix's protocols

(ns gpu.matrix
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]]
            [clojure.string :as str])
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

(defn all? 
  ([coll]
   (every? true? coll))
  ([f coll]
   (every? true? (mapv f coll))))

(defn in-bound? [^long idx ^long len]
  (and (>= idx 0) (< idx len)))

(defmacro with-check-indices [ndims m & args]
   (let [row (gensym "row_")
         col (gensym "col_")
         indexes (gensym "indexes_")
         check-clause
         (cond (= ndims 1)
               `(and (>= ~row 0)
                     (< ~row (first (mp/get-shape ~m))))
               (= ndims 2)
               `(let [shape# (mp/get-shape ~m)]
                 (and (>= ~row 0)
                      (< ~row (first shape#))
                      (>= ~col 0)
                      (< ~col (second shape#))))
               :else
               `(let [shape# (mp/get-shape ~m)] 
                  (all? (map-indexed
                          (fn [dim# idx#]
                            (and (>= idx# 0)
                                 (< idx# (nth shape# dim#))))
                          ~indexes))))
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
               (first args))]
    `(let ~(cond
             (= ndims 1)
             `[~row ~(first args)]
             (= ndims 2)
             `[~row ~(first args)
               ~col ~(second args)]
             :else 
             `[~indexes ~(first args)])
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
  `(if (= ~ndims (.dimensionality ~m))
     (do ~@body)
     (error
       (str "Invalid dimension! Expecting dimension " (.dimensionality ~m) " "
            "but got " ~ndims " instead"))))

(defmacro with-check-shape [ndims [m-sym m] & body]
  `(let [~(with-meta m-sym {:tag NDArray}) ~m]
     (with-check-ndims ~ndims ~m-sym
       (with-check-indices ~ndims ~m-sym
         ~@body))))

(extend-protocol mp/PIndexedAccess
  NDArray
  (get-1d [m ^long row]
    (with-check-shape
      1 [m m] row
      (.get m row)))
  (get-2d [m ^long row ^long col]
    (with-check-shape
      2 [m m] row col
      (.get m row col)))
  (get-nd [m indexes]
    (with-check-shape
      (count indexes) [m m] indexes
      (.get m (long-array indexes)))))

(extend-protocol mp/PIndexedSetting
  NDArray
  (set-1d [m ^long row ^double v]
    (with-check-shape
      1 [m m] row
      (.set (.clone m) row v)))
  (set-2d [m ^long row ^long col ^double v]
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
  (set-1d [m ^long row ^double v]
    (with-check-shape
      1 [m m] row
      (.set m row v)))
  (set-2d [m ^long row ^long col ^double v]
    (with-check-shape
      2 [m m] row col
      (.set m row col v)))
  (set-nd [m indexes ^double v]
    (with-check-shape
      (count indexes) [m m] indexes
      (.set m (long-array indexes) v))))

