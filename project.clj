(defproject gpu.matrix "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :jvm-opts [~(str "-Djava.library.path=src/native/:"
                   (System/getenv "$LD_LIBRARY_PATH"))]
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/core.matrix "0.40.0"]])
