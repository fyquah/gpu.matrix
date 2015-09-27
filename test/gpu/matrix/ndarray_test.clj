(ns gpu.matrix.ndarray-test
  (:require gpu.matrix
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.compliance-tester :as compliance]
            [clojure.test :refer :all])
  (:import gpu.matrix.NDArray))

(deftest compliance-tests-1D
  (compliance/instance-test (mp/construct-matrix (NDArray/sample) [ 1 2 3 4 5])))
