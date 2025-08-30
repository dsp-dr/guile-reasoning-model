;;; generation/kv-cache.scm - Key-Value cache for faster generation
(define-module (generation kv-cache)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-43)
  #:export (<kv-cache>
            make-kv-cache
            kv-cache-get
            kv-cache-update!
            kv-cache-reset!))

(define-record-type <kv-cache>
  (make-kv-cache-internal layers cache-data)
  kv-cache?
  (layers kv-cache-layers)
  (cache-data kv-cache-data))

(define (make-kv-cache n-layers)
  "Create a new KV cache for n-layers"
  (make-kv-cache-internal n-layers (make-vector n-layers #f)))

(define (kv-cache-get cache layer-idx)
  "Get cached values for layer"
  (vector-ref (kv-cache-data cache) layer-idx))

(define (kv-cache-update! cache layer-idx value)
  "Update cache for layer"
  (vector-set! (kv-cache-data cache) layer-idx value))

(define (kv-cache-reset! cache)
  "Reset all cache entries"
  (vector-fill! (kv-cache-data cache) #f))
