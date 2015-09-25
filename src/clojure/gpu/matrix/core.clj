(ns gpu.matrix.core
  (:require [gpu.matrix]
            [gpu.matrix.repl]
            [clojure.core.matrix.protocols :as mp])
  (:import gpu.matrix.KernelLoader
           gpu.matrix.NDArray))

(gpu.matrix.JVMLoader/init)
(gpu.matrix.repl/populate)

(defn -main []
  (let [m (mp/construct-matrix (NDArray/sample) [[4 0 3] [1 2 3]])
        v (mp/construct-matrix (NDArray/sample) [1 2 3])]
    (println "(+ m (+ m v))")
    (.print (mp/matrix-add m (mp/matrix-add m v)))))
