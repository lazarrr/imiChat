/// A single line of the trailing `Izvori:` block, parsed for display.
///
/// Backend lines look like:
///
///     [1] informatika_oas_2021.pdf — strana 12
///     [2] Osnovi programiranja — https://pmf.kg.ac.rs/...
///     [3] Program — strana 4 — https://pmf.kg.ac.rs/...?tab=1
class SourceCitation {
  const SourceCitation({required this.number, required this.label, this.url});

  /// The number inside the leading `[N]`, or null when the line has no prefix.
  final String? number;

  /// Human-readable citation without the URL (the URL is opened on tap).
  final String label;

  /// Web address of the source, when the backend supplied one. Local PDF/DOCX
  /// citations have no URL and are therefore not clickable.
  final Uri? url;
}

final RegExp _prefixPattern = RegExp(r'^\[(\d+)\]\s*(.*)$');
final RegExp _urlPattern = RegExp(r'https?://[^\s<>]+');
const String _trailingPunctuation = '.,;:!?';

/// Parses one raw citation line into its number, label and optional URL.
SourceCitation parseSourceCitation(String raw) {
  final trimmed = raw.trim();
  final prefix = _prefixPattern.firstMatch(trimmed);
  final number = prefix?.group(1);
  var label = (prefix?.group(2) ?? trimmed).trim();
  Uri? url;

  final match = _urlPattern.firstMatch(label);
  if (match != null) {
    final uri = _parseUrl(match.group(0)!);
    if (uri != null) {
      url = uri;
      label = label
          .replaceRange(match.start, match.end, '')
          .replaceFirst(RegExp(r'\s*[—–-]\s*$'), '')
          .trim();
    }
  }

  if (label.isEmpty && url != null) label = url.host;

  return SourceCitation(number: number, label: label, url: url);
}

Uri? _parseUrl(String raw) {
  var value = raw;
  while (value.isNotEmpty &&
      _trailingPunctuation.contains(value[value.length - 1])) {
    value = value.substring(0, value.length - 1);
  }

  final uri = Uri.tryParse(value);
  if (uri == null ||
      (uri.scheme != 'http' && uri.scheme != 'https') ||
      uri.host.isEmpty) {
    return null;
  }
  return uri;
}
