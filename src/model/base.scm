;;; model/base.scm - Base model interface
(define-module (model base)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-43)  ; vectors
  #:export (<model>
            make-model
            model?
            model-forward
            model-config
            model-eval-mode!))

(define-record-type <model>
  (make-model config forward-proc parameters)
  model?
  (config model-config)
  (forward-proc model-forward-proc)
  (parameters model-parameters))

(define (model-forward model input-ids #:key (cache #f))
  "Forward pass through model"
  ((model-forward-proc model) input-ids cache))

(define (model-eval-mode! model)
  "Set model to evaluation mode"
  ;; In a real implementation, this would disable dropout, etc.
  #t)
