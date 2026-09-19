// Proxies can return HTML error pages even when the application normally returns JSON.
export async function readApiResponse(response: Response) {
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    if (!response.ok)
      throw new Error(
        `Server unavailable (HTTP ${response.status}). Reconnecting may help; if this persists, check the service logs.`,
      );
    throw new Error("The server returned an unexpected response instead of JSON. Check the API proxy configuration.");
  }
  if (!response.ok)
    throw new Error(
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail) || `Request failed (HTTP ${response.status}).`,
    );
  return data;
}
