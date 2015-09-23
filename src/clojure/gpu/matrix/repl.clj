; Generates a useful repl environment
; Do not attempt to run this in runtime!
(ns gpu.matrix.repl)

(def code
  '(do
     (require '[clojure.core.matrix.protocols :as mp])
     (require '[clojure.core.matrix :as m])
     (require 'gpu.matrix)
     (def x 1)
     (def m (mp/construct-matrix  (gpu.matrix.NDArray.) [[1 2 3]]))
     (def arr (mp/construct-matrix (gpu.matrix.NDArray.) [1 2 3]))))

(defn populate []
  (eval code))

(populate)
