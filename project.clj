(defproject gpu.matrix "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/core.matrix "0.40.0"]]
  :aot [gpu.matrix.loader-utils]
  :prep-tasks  [["compile" "gpu.matrix.loader-utils"]
                "javac"]
  :profiles {:dev {:dependencies [[net.mikera/core.matrix.testing "0.0.4"]
                                  [net.mikera/vectorz-clj "0.30.1"]]
                   :jvm-opts [~(str "-Djava.library.path=src/native/:"
                                    (System/getenv "$LD_LIBRARY_PATH"))]}})

