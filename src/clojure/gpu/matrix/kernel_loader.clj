(ns gpu.matrix.kernel-loader
  (:require [clojure.java.io :as io])
  (:gen-class :name     gpu.matrix.KernelLoader
              :prefix   java- 
              :methods  [^{:static true} [loadProgram [String] String]
                         ^{:static true} [getIncludeProgramDir [] String]
                         ^{:static true} [loadLibrary [String] void]]))
(defmacro cond-starts-with [string & args]
  `(cond
     ~@(mapcat (fn [prefix ret]
                 [`(.startsWith ~string ~prefix) ret])
               (take-nth 2 args)
               (take-nth 2 (next args)))))

(def OPENCL-PREFIX "opencl/")
(def OPENCL-INCLUDE-DIR (.. ClassLoader (getSystemResource OPENCL-PREFIX) (getPath)))
(def SHARED-LIB-PATH
  (let [os-name (.. System (getProperty "os.name") (toLowerCase))
        os-arch (.. System (getProperty "os.arch"))]
    (str "native/"
         (cond-starts-with os-name
                           "win"      "win/"
                           "mac"      "mac/"
                           "linux"    "linux/"
                           "freebsd"  "freebsd"
                           "solaris"  "solaris/")
         os-arch "/")))

(defn get-lib-extension []
  (let [os-name (.. System (getProperty "os.name") (toLowerCase))]
    (str "." (cond-starts-with os-name
                               "win" "dll"
                               "mac" "dylib"
                               "linux" "so"))))

(defn ^{:static true}
  java-loadProgram
  ^String [^String file-name]
  (slurp (ClassLoader/getSystemResource (str OPENCL-PREFIX file-name))))

(defn ^{:static true}
  java-getIncludeProgramDir
  ^String []
  (str "-I " OPENCL-INCLUDE-DIR))

(def libs-loaded-atom (atom #{}))

(defn ^{:static true}
  java-loadLibrary
  "Attempt to load from java.library.path where possible, otherwise load
  from resources/~SHARED-LIB-PATH"
  [^String lib-name]
  (when-not (contains? @libs-loaded-atom lib-name)
    (try
      (System/loadLibrary lib-name)
      (swap! libs-loaded-atom conj lib-name)
      (catch UnsatisfiedLinkError e
        (let [lib-extension (get-lib-extension) 
              ; ^ not exactly necessary, but for elegance sake 
              tmp-dir (System/getProperty "java.io.tmpdir")
              current-time (.. (new java.util.Date) (getTime)) 
              tmp-lib-path (str tmp-dir
                                current-time ; to prevent duplication
                                "lib" lib-name lib-extension)]
          (with-open [in (ClassLoader/getSystemResourceAsStream
                           (str SHARED-LIB-PATH "lib" lib-name lib-extension))]
            (with-open [out (io/output-stream tmp-lib-path)]
              (io/copy in out)))
          (System/load tmp-lib-path)
          (swap! libs-loaded-atom conj lib-name))))))

