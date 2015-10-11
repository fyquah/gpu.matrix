; Specialized Real Vector implementation
; based on the class gpu.matrix.Vector

(ns gpu.matrix.vector
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.utils :refer [error]]
            [clojure.string :as str])
  (:import gpu.matrix.NDArray gpu.matrix.Initializer))

