(ns gpu.matrix.core
  (:require gpu.matrix
            gpu.matrix.repl)
  (:import gpu.matrix.KernelLoader))

(gpu.matrix.JVMLoader/init)
(gpu.matrix.repl/populate)

(defn -main []
  (.print (.add m (.add m m))))
