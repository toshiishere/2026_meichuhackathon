import React, { useEffect, useRef, useState } from "react";
export function RemoveSessionDialog({
  sessionId,
  onCancel,
  onRemove,
  busy,
  error,
}: {
  sessionId: string;
  onCancel: () => void;
  onRemove: () => void;
  busy: boolean;
  error: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [confirmation, setConfirmation] = useState("");
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="remove-dialog"
      onCancel={onCancel}
      aria-labelledby="remove-session-title"
    >
      <h2 id="remove-session-title">Remove session permanently?</h2>
      <p>
        This deletes the CSI, video, frame timestamps, metadata, and logs for{" "}
        <strong>{sessionId}</strong>. This cannot be undone.
      </p>
      {error && (
        <div role="alert" className="alert">
          {error}
        </div>
      )}
      <label className="field">
        <span>Type the session ID to confirm</span>
        <input
          aria-label="Type the session ID to confirm"
          autoFocus
          value={confirmation}
          onChange={(e) => setConfirmation(e.target.value)}
        />
      </label>
      <div className="actions">
        <button onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <button
          className="danger"
          disabled={busy || confirmation !== sessionId}
          onClick={onRemove}
        >
          Permanently remove
        </button>
      </div>
    </dialog>
  );
}
