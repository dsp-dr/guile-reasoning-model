;;; generation/basic.scm - Basic text generation
(define-module (generation basic)
  #:use-module (tokenizer base)
  #:use-module (model base)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-43)
  #:export (generate-text-basic
            generate-text-with-cache))

(define (argmax vec)
  "Return index of maximum value in vector"
  (let ((max-val (vector-ref vec 0))
        (max-idx 0))
    (do ((i 1 (+ i 1)))
        ((>= i (vector-length vec)) max-idx)
      (when (> (vector-ref vec i) max-val)
        (set! max-val (vector-ref vec i))
        (set! max-idx i)))))

(define* (generate-text-basic model tokenizer prompt 
                              #:key (max-new-tokens 100) (eos-token-id #f))
  "Basic sequential text generation"
  (let* ((input-ids (encode tokenizer prompt))
         (generated '()))
    
    (model-eval-mode! model)
    
    (do ((i 0 (+ i 1))
         (current-ids input-ids))
        ((or (>= i max-new-tokens)
             (and eos-token-id 
                  (not (null? generated))
                  (= (car generated) eos-token-id)))
         (decode tokenizer (reverse generated)))
      
      (let* ((logits (model-forward model current-ids))
             (last-logits (vector-ref logits (- (vector-length logits) 1)))
             (next-token (argmax last-logits)))
        (set! generated (cons next-token generated))
        (set! current-ids (append current-ids (list next-token)))))))
