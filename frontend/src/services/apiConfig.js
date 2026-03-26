export const PROD_API_URL = "https://rooftop-solar-detection.onrender.com";

export function resolveApiBaseUrl(env) {
  const explicitUrl = env?.VITE_API_URL?.trim();
  if (explicitUrl) {
    return explicitUrl.replace(/\/+$/, "");
  }

  return env?.DEV ? "/api" : PROD_API_URL;
}
