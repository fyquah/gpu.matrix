; This is the main interface to exntending core.matrix's protocols

(ns gpu.matrix
  (:require [clojure.core.matrix.protocols :as mp])
  (:import gpu.matrix.NDArray))

(extend-protocol mp/PImplementation
  NDArray
  (implementation-key [m] :gpu-matrix)
  (meta-info [m] {})
  (construct-matrix [m data]
    ; TODO
    (let [ndims (mp/dimensionality data)]
      ))
  (new-vector [m length]
    ; TODO
    )
  (new-matrix [m rows columns]
    ; TODO
    )
  (new-matrix-nd [m shape]
    ; TODO
    )
  (supports-dimensionality? [m dimensions]
    true))
