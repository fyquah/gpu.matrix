(ns gpu.matrix.ndarray-benchmark
  (:require [criterium.core :as c]
            [clojure.core.matrix :as m]
            [gpu.matrix]))

(def sample-data (m/to-nested-vectors (repeat 2 (range 2))))
(def bench-data (atom {}))

(defn benchmark []
  (println "Running gpu.matrix")
  (let [m (m/matrix (gpu.matrix.NDArray/sample) sample-data)]
    (swap! bench-data
           assoc :gpu.matrix
           (c/bench (m/add m m))))
  (println "Benchmarking vectorz")
  (let [m (m/matrix :vectorz sample-data)]
    (swap! bench-data
           assoc :vectorz
           (c/bench (m/add m m)))))

(benchmark)
