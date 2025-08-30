;;; Text generation with pre-trained models
;;; Chapter 2 implementation

(define-module (generation text-gen)
  #:use-module (ice-9 match)
  #:use-module (ice-9 format)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-43)  ; vectors
  #:export (make-tokenizer
            tokenize
            detokenize
            generate-text
            make-generation-config
            token->id
            id->token))

;;; Tokenizer implementation

(define-record-type <tokenizer>
  (make-tokenizer* vocab special-tokens)
  tokenizer?
  (vocab tokenizer-vocab)
  (special-tokens tokenizer-special-tokens))

(define* (make-tokenizer #:key (vocab-size 10000))
  "Create a simple tokenizer with basic vocabulary."
  (let ((vocab (make-hash-table))
        (special-tokens '(("<PAD>" . 0)
                         ("<UNK>" . 1)
                         ("<BOS>" . 2)
                         ("<EOS>" . 3)
                         ("<THINK>" . 4)
                         ("<STEP>" . 5))))
    ;; Initialize special tokens
    (for-each (lambda (pair)
               (hash-set! vocab (car pair) (cdr pair)))
             special-tokens)
    (make-tokenizer* vocab special-tokens)))

(define (tokenize tokenizer text)
  "Convert text to token IDs."
  (let ((vocab (tokenizer-vocab tokenizer))
        (words (string-split text #\space)))
    (map (lambda (word)
          (hash-ref vocab word 
                   (hash-ref vocab "<UNK>")))
        words)))

(define (detokenize tokenizer ids)
  "Convert token IDs back to text."
  (let ((reverse-vocab (make-hash-table)))
    ;; Build reverse vocabulary
    (hash-for-each (lambda (k v)
                    (hash-set! reverse-vocab v k))
                  (tokenizer-vocab tokenizer))
    (string-join 
     (map (lambda (id)
           (hash-ref reverse-vocab id "<UNK>"))
         ids)
     " ")))

(define (token->id tokenizer token)
  "Get ID for a specific token."
  (hash-ref (tokenizer-vocab tokenizer) token
           (hash-ref (tokenizer-vocab tokenizer) "<UNK>")))

(define (id->token tokenizer id)
  "Get token for a specific ID."
  (let ((reverse-vocab (make-hash-table)))
    (hash-for-each (lambda (k v)
                    (hash-set! reverse-vocab v k))
                  (tokenizer-vocab tokenizer))
    (hash-ref reverse-vocab id "<UNK>")))

;;; Generation configuration

(define-record-type <generation-config>
  (make-generation-config* max-tokens temperature top-p top-k)
  generation-config?
  (max-tokens config-max-tokens)
  (temperature config-temperature)
  (top-p config-top-p)
  (top-k config-top-k))

(define* (make-generation-config #:key 
                                 (max-tokens 100)
                                 (temperature 0.7)
                                 (top-p 0.9)
                                 (top-k 50))
  "Create configuration for text generation."
  (make-generation-config* max-tokens temperature top-p top-k))

;;; Text generation engine

(define* (generate-text prompt 
                       #:key 
                       (model-fn simple-language-model)
                       (tokenizer (make-tokenizer))
                       (config (make-generation-config)))
  "Generate text from a prompt using the specified model."
  (let* ((input-ids (tokenize tokenizer prompt))
         (max-tokens (config-max-tokens config))
         (temperature (config-temperature config)))
    (let loop ((current-ids input-ids)
               (generated '())
               (steps 0))
      (if (>= steps max-tokens)
          (detokenize tokenizer (append input-ids (reverse generated)))
          (let* ((next-token-probs (model-fn current-ids))
                 (next-token-id (sample-token next-token-probs temperature))
                 (eos-id (token->id tokenizer "<EOS>")))
            (if (= next-token-id eos-id)
                (detokenize tokenizer (append input-ids (reverse generated)))
                (loop (append current-ids (list next-token-id))
                      (cons next-token-id generated)
                      (1+ steps))))))))

;;; Simple language model (placeholder)

(define (simple-language-model input-ids)
  "Simplified language model for demonstration."
  ;; Return random probability distribution
  (let ((vocab-size 10000))
    (list->vector 
     (map (lambda (_) (random:uniform))
          (iota vocab-size)))))

;;; Sampling strategies

(define* (sample-token probs #:optional (temperature 1.0))
  "Sample a token from probability distribution."
  (let* ((scaled-probs (apply-temperature probs temperature))
         (cumsum (cumulative-sum scaled-probs))
         (rand (random:uniform)))
    (vector-index (lambda (cs) (> cs rand)) cumsum)))

(define (apply-temperature probs temperature)
  "Apply temperature scaling to probabilities."
  (if (= temperature 0)
      probs  ; Greedy decoding
      (let* ((log-probs (vector-map (lambda (i p)
                                      (/ (log (max p 1e-10)) 
                                         temperature))
                                    probs))
             (max-log (vector-fold (lambda (i acc p) (max acc p))
                                  -inf.0 log-probs))
             (exp-probs (vector-map (lambda (i p)
                                      (exp (- p max-log)))
                                    log-probs))
             (sum (vector-fold (lambda (i acc p) (+ acc p))
                              0 exp-probs)))
        (vector-map (lambda (i p) (/ p sum)) exp-probs))))

(define (cumulative-sum vec)
  "Calculate cumulative sum of vector elements."
  (let ((result (make-vector (vector-length vec))))
    (vector-set! result 0 (vector-ref vec 0))
    (do ((i 1 (1+ i)))
        ((>= i (vector-length vec)) result)
      (vector-set! result i 
                   (+ (vector-ref result (1- i))
                      (vector-ref vec i))))))

(define (vector-index pred vec)
  "Find first index where predicate is true."
  (let loop ((i 0))
    (cond ((>= i (vector-length vec)) #f)
          ((pred (vector-ref vec i)) i)
          (else (loop (1+ i))))))

;;; Caching for efficient generation

(define-record-type <kv-cache>
  (make-kv-cache keys values)
  kv-cache?
  (keys kv-cache-keys set-kv-cache-keys!)
  (values kv-cache-values set-kv-cache-values!))

(define (create-kv-cache)
  "Create key-value cache for attention."
  (make-kv-cache (make-hash-table) (make-hash-table)))

(define (update-cache! cache key value)
  "Update cache with new key-value pair."
  (hash-set! (kv-cache-keys cache) key value)
  value)

(define (get-cached cache key)
  "Retrieve cached value or #f."
  (hash-ref (kv-cache-keys cache) key #f))