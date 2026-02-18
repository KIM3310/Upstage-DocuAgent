(function () {
  var DEFAULT_GA = "G-XXXXXXXXXX";
  var DEFAULT_CLARITY = "CLARITY_PROJECT_ID";
  var DEFAULT_GTM = "GTM-MHK4C4D7";

  function readMeta(name) {
    var node = document.querySelector("meta[name=\"" + name + "\"]");
    return node ? (node.getAttribute("content") || "").trim() : "";
  }

  function isLocalhost() {
    return location.hostname === "localhost" || location.hostname === "127.0.0.1";
  }

  function isDntEnabled() {
    return (
      navigator.doNotTrack === "1" ||
      window.doNotTrack === "1" ||
      navigator.msDoNotTrack === "1" ||
      navigator.globalPrivacyControl === true
    );
  }

  function hasConsent(requireConsent) {
    if (!requireConsent) return true;
    try {
      return localStorage.getItem("analytics_consent") === "granted";
    } catch (_err) {
      return false;
    }
  }

  function validGaId(value) {
    if (!value) return false;
    return /^G-[A-Z0-9]{6,}$/i.test(value) && !/x{4,}/i.test(value);
  }

  function validClarityId(value) {
    if (!value) return false;
    if (/clarity|xxxx|placeholder/i.test(value)) return false;
    return /^[a-z0-9]{6,24}$/i.test(value);
  }

  function validGtmId(value) {
    if (!value) return false;
    return /^GTM-[A-Z0-9]{6,}$/i.test(value);
  }

  function resolveGaId() {
    var override = (window.__GA_MEASUREMENT_ID__ || "").trim();
    if (validGaId(override)) return override;

    var meta = (readMeta("ga-measurement-id") || "").trim();
    if (validGaId(meta)) return meta;

    var fallback = (DEFAULT_GA || "").trim();
    return validGaId(fallback) ? fallback : "";
  }

  function resolveClarityId() {
    var override = (window.__CLARITY_PROJECT_ID__ || "").trim();
    if (validClarityId(override)) return override;

    var meta = (readMeta("clarity-project-id") || "").trim();
    if (validClarityId(meta)) return meta;

    var fallback = (DEFAULT_CLARITY || "").trim();
    return validClarityId(fallback) ? fallback : "";
  }

  function resolveGtmId() {
    var override = (window.__GTM_CONTAINER_ID__ || "").trim();
    if (validGtmId(override)) return override;

    var meta = (readMeta("gtm-container-id") || "").trim();
    if (validGtmId(meta)) return meta;

    var fallback = (DEFAULT_GTM || "").trim();
    return validGtmId(fallback) ? fallback : "";
  }

  function ensureScript(id, src) {
    if (document.getElementById(id)) return;
    var script = document.createElement("script");
    script.id = id;
    script.async = true;
    script.src = src;
    document.head.appendChild(script);
  }

  function hasGtmInstalled() {
    if (window.google_tag_manager) return true;
    return !!document.querySelector("script[src*='googletagmanager.com/gtm.js']");
  }

  function enableGtm(gtmId, consentGranted) {
    if (!validGtmId(gtmId) || hasGtmInstalled()) return;

    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({
      "gtm.start": new Date().getTime(),
      event: "gtm.js",
      analytics_storage: consentGranted ? "granted" : "denied",
      ad_storage: "denied",
      ad_user_data: "denied",
      ad_personalization: "denied"
    });

    ensureScript("gtm-script", "https://www.googletagmanager.com/gtm.js?id=" + encodeURIComponent(gtmId));
  }

  function enableGa(gaId, consentGranted) {
    ensureScript("ga4-script", "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(gaId));
    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
    window.gtag("js", new Date());
    window.gtag("consent", "default", {
      analytics_storage: consentGranted ? "granted" : "denied",
      ad_storage: "denied",
      ad_user_data: "denied",
      ad_personalization: "denied",
      functionality_storage: "granted",
      security_storage: "granted"
    });
    window.gtag("config", gaId, {
      anonymize_ip: true,
      allow_google_signals: false,
      allow_ad_personalization_signals: false,
      transport_type: "beacon",
      page_path: location.pathname + location.search
    });
  }

  function enableClarity(clarityId) {
    if (window.clarity) return;
    (function (c, l, a, r, i, t, y) {
      c[a] = c[a] || function () { (c[a].q = c[a].q || []).push(arguments); };
      t = l.createElement(r);
      t.async = 1;
      t.src = "https://www.clarity.ms/tag/" + i;
      y = l.getElementsByTagName(r)[0];
      y.parentNode.insertBefore(t, y);
    })(window, document, "clarity", "script", clarityId);
  }

  var gaId = resolveGaId();
  var clarityId = resolveClarityId();
  var gtmId = resolveGtmId();

  var consentMeta = readMeta("analytics-require-consent") || "false";
  var consentOverride = window.__ANALYTICS_REQUIRE_CONSENT__;
  var requireConsent = typeof consentOverride === "boolean"
    ? consentOverride
    : /^true$/i.test((consentOverride || consentMeta).toString());

  if (isLocalhost() || isDntEnabled()) {
    return;
  }

  var consentGranted = hasConsent(requireConsent);
  if (!consentGranted) {
    return;
  }

  if (validGtmId(gtmId)) {
    enableGtm(gtmId, consentGranted);
  }

  if (validGaId(gaId) && !hasGtmInstalled()) {
    enableGa(gaId, consentGranted);
  }

  if (validClarityId(clarityId) && !hasGtmInstalled()) {
    enableClarity(clarityId);
  }
})();
