export type PhoneBufferMeta = {
  id: string;
  settings: { width: number; height: number; fps: number; name: string };
  origin: number;
  nextSequence: number;
  count: number;
  bytes: number;
  resumeToken?: string;
  paused: boolean;
  skipped: number;
};
type StoredFrame = {
  id: string;
  sequence: number;
  payload: Blob;
  bytes: number;
};
const LIMIT = 512 * 1024 * 1024;
const request = <T>(r: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
async function database() {
  const r = indexedDB.open("csi-phone-buffer", 1);
  r.onupgradeneeded = () => {
    r.result.createObjectStore("streams", { keyPath: "id" });
    r.result.createObjectStore("frames", { keyPath: ["id", "sequence"] });
  };
  return request(r);
}
export async function pendingPhoneStreams(): Promise<PhoneBufferMeta[]> {
  const db = await database();
  try {
    return await request(
      db.transaction("streams").objectStore("streams").getAll(),
    );
  } finally {
    db.close();
  }
}
export class PhoneBuffer {
  private chain: Promise<unknown> = Promise.resolve();
  private constructor(
    private db: IDBDatabase,
    public meta: PhoneBufferMeta,
  ) {}
  static async open(id: string, settings?: PhoneBufferMeta["settings"]) {
    const db = await database();
    let meta = (await request(
      db.transaction("streams").objectStore("streams").get(id),
    )) as PhoneBufferMeta | undefined;
    if (!meta) {
      if (!settings) {
        db.close();
        throw new Error("No saved phone upload was found.");
      }
      meta = {
        id,
        settings,
        origin: performance.timeOrigin,
        nextSequence: 0,
        count: 0,
        bytes: 0,
        paused: false,
        skipped: 0,
      };
    }
    const buffer = new PhoneBuffer(db, meta);
    await buffer.update({});
    return buffer;
  }
  now() {
    return performance.timeOrigin + performance.now() - this.meta.origin;
  }
  private mutate<T>(
    operation: (
      tx: IDBTransaction,
      meta: PhoneBufferMeta,
      resolve: (value: T) => void,
    ) => void,
  ): Promise<T> {
    const run = () =>
      new Promise<T>((resolve, reject) => {
        const tx = this.db.transaction(["streams", "frames"], "readwrite");
        const meta = { ...this.meta };
        let result: T;
        tx.oncomplete = () => {
          this.meta = meta;
          resolve(result);
        };
        tx.onabort = () =>
          reject(tx.error || new Error("Phone storage transaction failed"));
        try {
          operation(tx, meta, (value) => {
            result = value;
          });
        } catch (error) {
          tx.abort();
          reject(error);
        }
      });
    const result = this.chain.then(run);
    this.chain = result.catch(() => {});
    return result;
  }
  update(changes: Partial<PhoneBufferMeta>) {
    return this.mutate<void>((tx, meta, done) => {
      Object.assign(meta, changes);
      tx.objectStore("streams").put(meta);
      done();
    });
  }
  append(jpeg: Blob, captureMs: number, skipped: number) {
    return this.mutate<number>((tx, meta, done) => {
      if (meta.bytes + jpeg.size + 20 > LIMIT)
        throw new Error(
          "The 512 MiB phone upload buffer is full. Capture is paused; keep this page open to upload saved frames.",
        );
      const sequence = meta.nextSequence++;
      const header = new ArrayBuffer(20),
        view = new DataView(header);
      view.setUint32(0, 0x43534931);
      view.setUint32(4, sequence);
      view.setFloat64(8, captureMs);
      view.setUint32(16, skipped);
      const payload = new Blob([header, jpeg]);
      tx.objectStore("frames").add({
        id: meta.id,
        sequence,
        payload,
        bytes: payload.size,
      });
      meta.count++;
      meta.bytes += payload.size;
      meta.skipped = skipped;
      tx.objectStore("streams").put(meta);
      done(sequence);
    });
  }
  acknowledge(sequence: number) {
    if (sequence < 0) return Promise.resolve();
    return this.mutate<void>((tx, meta, done) => {
      const cursor = tx
        .objectStore("frames")
        .openCursor(IDBKeyRange.bound([meta.id, 0], [meta.id, sequence]));
      cursor.onsuccess = () => {
        const c = cursor.result;
        if (c) {
          meta.count--;
          meta.bytes -= (c.value as StoredFrame).bytes;
          c.delete();
          c.continue();
        } else {
          tx.objectStore("streams").put(meta);
          done();
        }
      };
    });
  }
  async batch(after: number, limit: number): Promise<StoredFrame[]> {
    await this.chain;
    return request(
      this.db
        .transaction("frames")
        .objectStore("frames")
        .getAll(
          IDBKeyRange.bound(
            [this.meta.id, after + 1],
            [this.meta.id, Number.MAX_SAFE_INTEGER],
          ),
          limit,
        ),
    );
  }
  async complete() {
    await this.chain;
    if (this.meta.count) throw new Error("Phone still has buffered frames");
    await new Promise<void>((resolve, reject) => {
      const tx = this.db.transaction("streams", "readwrite");
      tx.objectStore("streams").delete(this.meta.id);
      tx.oncomplete = () => resolve();
      tx.onabort = () => reject(tx.error);
    });
  }
  async close() {
    await this.chain;
    this.db.close();
  }
}
