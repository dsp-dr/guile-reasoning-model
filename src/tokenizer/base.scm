;;; tokenizer/base.scm - Basic tokenizer interface
(define-module (tokenizer base)
  #:use-module (ice-9 match)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:export (<tokenizer>
            make-tokenizer
            tokenizer?
            encode
            decode
            tokenizer-vocab-size
            tokenizer-eos-token-id))

(define-record-type <tokenizer>
  (make-tokenizer vocab encode-proc decode-proc special-tokens)
  tokenizer?
  (vocab tokenizer-vocab)
  (encode-proc tokenizer-encode-proc)
  (decode-proc tokenizer-decode-proc)
  (special-tokens tokenizer-special-tokens))

(define (encode tokenizer text)
  "Encode text to token IDs"
  ((tokenizer-encode-proc tokenizer) text))

(define (decode tokenizer token-ids)
  "Decode token IDs to text"
  ((tokenizer-decode-proc tokenizer) token-ids))

(define (tokenizer-vocab-size tokenizer)
  "Get vocabulary size"
  (hash-count (const #t) (tokenizer-vocab tokenizer)))

(define (tokenizer-eos-token-id tokenizer)
  "Get end-of-sequence token ID"
  (hash-ref (tokenizer-special-tokens tokenizer) 'eos))
