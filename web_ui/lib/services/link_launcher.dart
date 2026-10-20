import 'package:url_launcher/url_launcher.dart';

/// Opens a single external link. Injected in tests so taps can be observed
/// without touching the browser.
typedef LinkLauncher = Future<bool> Function(Uri url);

/// Validates and opens [url] in a new browser tab.
///
/// Only `http` and `https` URLs are allowed — chat content is untrusted input,
/// so `javascript:`, `file:` and other schemes must never reach the browser.
/// Returns `false` when the URL is rejected or the platform cannot open it
/// instead of throwing into the widget tree.
Future<bool> openExternalLink(Uri url, {LinkLauncher? launcher}) async {
  if ((url.scheme != 'http' && url.scheme != 'https') || url.host.isEmpty) {
    return false;
  }

  try {
    return await (launcher ?? _launch)(url);
  } catch (_) {
    return false;
  }
}

/// Opens a markdown `href` (which may be null or malformed).
Future<bool> openExternalHref(String? href, {LinkLauncher? launcher}) {
  final uri = href == null ? null : Uri.tryParse(href);
  if (uri == null) return Future.value(false);
  return openExternalLink(uri, launcher: launcher);
}

Future<bool> _launch(Uri url) => launchUrl(url, webOnlyWindowName: '_blank');
