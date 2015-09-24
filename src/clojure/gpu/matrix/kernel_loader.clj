(ns gpu.matrix.kernel-loader
  (:gen-class :name     gpu.matrix.KernelLoader
              :prefix   java- 
              :methods  [^{:static true} [loadProgram [String] String]
                         ^{:static true} [getIncludeProgramDir [] String]]))

(def OPENCL-PREFIX "opencl/")
(def OPENCL-INCLUDE-DIR (.. ClassLoader (getSystemResource OPENCL-PREFIX) (getPath)))

(defn ^{:static true}
  java-loadProgram
  ^String [^String file-name]
  (slurp (ClassLoader/getSystemResource (str OPENCL-PREFIX file-name))))

(defn ^{:static true}
  java-getIncludeProgramDir
  ^String []
  (str "-I " OPENCL-INCLUDE-DIR))
