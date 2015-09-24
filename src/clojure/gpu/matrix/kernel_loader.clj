(ns gpu.matrix.kernel-loader
  (:gen-class :name     gpu.matrix.KernelLoader
              :prefix   java- 
              :methods  [^{:static true} [load-file [String] String]]))

(def OPENCL-PREFIX "opencl/")

(defn ^{:static true} load-file ^String [^String file-name]
  (ClassLoader/getSystemResource (str OPENCL-PREFIX file-name)))

