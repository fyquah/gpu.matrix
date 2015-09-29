(ns gpu.matrix.ndarray-test
  (:require gpu.matrix
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.compliance-tester :as compliance]
            [clojure.test :refer :all])
  (:import gpu.matrix.NDArray))

(comment
  (deftest compliance-tests-1D
    (println "1D")
    (compliance/instance-test (mp/construct-matrix (NDArray/sample) [ 1 2 3 4 5]))
    (println "Finishsed 1d")))

(deftest compliance-tests-2D
  (let [m (mp/construct-matrix (NDArray/sample)
                               [[1 2 3 4 5]
                                [6 7 8 9 10]
                                [11 12 13 14 15]
                                [16 17 18 19 20]
                                [21 22 23 24 25]])]
    (compliance/instance-test m)))

