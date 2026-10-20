import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../models/chat_message.dart';
import '../models/conversation.dart';
import '../models/source_citation.dart';
import '../services/link_launcher.dart';
import '../theme.dart';

/// A single chat turn.
///
/// Newly created messages fade + slide in; messages restored from storage or
/// scrolled back into view render statically so the list never "replays"
/// itself.
class MessageBubble extends StatelessWidget {
  const MessageBubble({super.key, required this.message, this.onOpenLink});

  final ChatMessage message;

  /// Overrides how links are opened — tests use this to record taps instead of
  /// launching a browser.
  final LinkLauncher? onOpenLink;

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == MessageRole.user;
    final isFresh = DateTime.now().difference(message.timestamp).inSeconds < 2;

    final content = isUser
        ? _UserMessage(message: message)
        : _AssistantMessage(message: message, onOpenLink: onOpenLink);

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: isFresh ? 0.0 : 1.0, end: 1.0),
      duration: const Duration(milliseconds: 300),
      curve: kEase,
      builder: (context, t, child) => Opacity(
        opacity: t,
        child: Transform.translate(
          offset: Offset(0, 12 * (1 - t)),
          child: child,
        ),
      ),
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: isUser ? 9 : 12),
        child: content,
      ),
    );
  }
}

// --------------------------------------------------------------- user side -- //

class _UserMessage extends StatelessWidget {
  const _UserMessage({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.end,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Flexible(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 620),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 17, vertical: 12),
              decoration: BoxDecoration(
                gradient: kBrandGradient,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(20),
                  topRight: Radius.circular(20),
                  bottomLeft: Radius.circular(20),
                  bottomRight: Radius.circular(6),
                ),
                boxShadow: [
                  BoxShadow(
                    color: kIndigo.withValues(alpha: 0.28),
                    blurRadius: 18,
                    spreadRadius: -4,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: SelectableText(
                message.content,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14.5,
                  height: 1.5,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(width: 44),
      ],
    );
  }
}

// ---------------------------------------------------------- assistant side -- //

class _AssistantMessage extends StatelessWidget {
  const _AssistantMessage({required this.message, this.onOpenLink});

  final ChatMessage message;
  final LinkLauncher? onOpenLink;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _AssistantAvatar(),
        const SizedBox(width: 12),
        Flexible(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _MetaRow(message: message),
              const SizedBox(height: 7),
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 760),
                child: _AssistantBubble(
                  message: message,
                  onOpenLink: onOpenLink,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _MetaRow extends StatelessWidget {
  const _MetaRow({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    final time = TimeOfDay.fromDateTime(message.timestamp).format(context);

    return Row(
      children: [
        Text(
          'UniChat',
          style: TextStyle(
            fontSize: 12.5,
            fontWeight: FontWeight.w700,
            letterSpacing: -0.1,
            color: p.textPrimary,
          ),
        ),
        const SizedBox(width: 7),
        Container(
          width: 3,
          height: 3,
          decoration: BoxDecoration(color: p.textMuted, shape: BoxShape.circle),
        ),
        const SizedBox(width: 7),
        Text(time, style: TextStyle(fontSize: 11.5, color: p.textMuted)),
        if (message.content.trim().isNotEmpty) ...[
          const SizedBox(width: 6),
          _CopyAction(text: message.content),
        ],
      ],
    );
  }
}

class _CopyAction extends StatefulWidget {
  const _CopyAction({required this.text});

  final String text;

  @override
  State<_CopyAction> createState() => _CopyActionState();
}

class _CopyActionState extends State<_CopyAction> {
  bool _copied = false;
  bool _hovering = false;

  Future<void> _copy() async {
    await Clipboard.setData(ClipboardData(text: widget.text));
    if (!mounted) return;
    setState(() => _copied = true);
    await Future<void>.delayed(const Duration(milliseconds: 1400));
    if (!mounted) return;
    setState(() => _copied = false);
  }

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    final color = _copied
        ? kEmerald
        : (_hovering ? p.textSecondary : p.textMuted);

    return Tooltip(
      message: _copied ? 'Kopirano' : 'Kopiraj odgovor',
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        onEnter: (_) => setState(() => _hovering = true),
        onExit: (_) => setState(() => _hovering = false),
        child: GestureDetector(
          onTap: _copy,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 160),
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
            decoration: BoxDecoration(
              color: _hovering ? p.hover : Colors.transparent,
              borderRadius: BorderRadius.circular(R.xs - 2),
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 180),
              child: Icon(
                _copied ? Icons.check_rounded : Icons.copy_rounded,
                key: ValueKey(_copied),
                size: 14,
                color: color,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _AssistantAvatar extends StatelessWidget {
  const _AssistantAvatar();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 34,
      height: 34,
      decoration: BoxDecoration(
        gradient: kBrandGradient,
        borderRadius: BorderRadius.circular(R.sm),
        boxShadow: [
          BoxShadow(
            color: kIndigo.withValues(alpha: 0.32),
            blurRadius: 12,
            spreadRadius: -3,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: const Icon(
        Icons.auto_awesome_rounded,
        color: Colors.white,
        size: 17,
      ),
    );
  }
}

class _AssistantBubble extends StatelessWidget {
  const _AssistantBubble({required this.message, this.onOpenLink});

  final ChatMessage message;
  final LinkLauncher? onOpenLink;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    final result = splitSources(message.content);

    return Container(
      padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
      decoration: BoxDecoration(
        color: p.assistantBubble,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(6),
          topRight: Radius.circular(20),
          bottomLeft: Radius.circular(20),
          bottomRight: Radius.circular(20),
        ),
        border: Border.all(color: p.border),
        boxShadow: [
          BoxShadow(
            color: p.shadow,
            blurRadius: 18,
            spreadRadius: -8,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (result.text.isEmpty && message.streaming)
            const _TypingIndicator()
          else
            // Markdown's `selectable` mode renders through a hidden text
            // field, which swallows link taps on the web. SelectionArea keeps
            // the answer selectable while links stay plain text spans that
            // receive pointer events normally.
            SelectionArea(
              child: MarkdownBody(
                data: result.text,
                styleSheet: _markdownStyle(context, p),
                onTapLink: (text, href, title) {
                  unawaited(openExternalHref(href, launcher: onOpenLink));
                },
              ),
            ),
          if (message.streaming && result.text.isNotEmpty)
            const _StreamingCursor(),
          if (result.sources.isNotEmpty) ...[
            const SizedBox(height: 14),
            _SourcesCard(sources: result.sources, onOpenLink: onOpenLink),
          ],
        ],
      ),
    );
  }
}

MarkdownStyleSheet _markdownStyle(BuildContext context, UniPalette p) {
  final body = TextStyle(fontSize: 14.5, height: 1.65, color: p.textPrimary);

  return MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
    p: body,
    strong: body.copyWith(fontWeight: FontWeight.w700),
    em: body.copyWith(fontStyle: FontStyle.italic),
    h1: body.copyWith(fontSize: 19, fontWeight: FontWeight.w800, height: 1.35),
    h2: body.copyWith(fontSize: 17, fontWeight: FontWeight.w800, height: 1.35),
    h3: body.copyWith(
      fontSize: 15.5,
      fontWeight: FontWeight.w700,
      height: 1.35,
    ),
    a: body.copyWith(
      color: Theme.of(context).colorScheme.primary,
      decoration: TextDecoration.underline,
      decorationColor: Theme.of(context).colorScheme.primary,
    ),
    blockquoteDecoration: BoxDecoration(
      color: kIndigo.withValues(alpha: p.isDark ? 0.16 : 0.06),
      borderRadius: BorderRadius.circular(R.xs),
      border: const Border(left: BorderSide(color: kIndigo, width: 3)),
    ),
    blockquotePadding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
    code: TextStyle(
      fontFamily: 'monospace',
      backgroundColor: p.surfaceSunken,
      fontSize: 13,
      color: p.textPrimary,
    ),
    codeblockDecoration: BoxDecoration(
      color: p.surfaceSunken,
      borderRadius: BorderRadius.circular(R.sm),
      border: Border.all(color: p.border),
    ),
    codeblockPadding: const EdgeInsets.all(14),
    listBullet: body,
    listIndent: 22,
    horizontalRuleDecoration: BoxDecoration(
      border: Border(top: BorderSide(color: p.border)),
    ),
    tableBorder: TableBorder.all(
      color: p.border,
      borderRadius: BorderRadius.circular(R.xs),
    ),
    tableCellsPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
  );
}

// ---------------------------------------------------------------- sources -- //

class _SourcesCard extends StatelessWidget {
  const _SourcesCard({required this.sources, this.onOpenLink});

  final List<String> sources;
  final LinkLauncher? onOpenLink;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(12, 11, 12, 12),
      decoration: BoxDecoration(
        color: p.surfaceSunken,
        borderRadius: BorderRadius.circular(R.md),
        border: Border.all(color: p.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.link_rounded,
                size: 14,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(width: 6),
              Text(
                'Izvori',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.8,
                  color: p.textSecondary,
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 7,
                  vertical: 1.5,
                ),
                decoration: BoxDecoration(
                  color: kIndigo.withValues(alpha: p.isDark ? 0.28 : 0.12),
                  borderRadius: BorderRadius.circular(R.pill),
                ),
                child: Text(
                  '${sources.length}',
                  style: TextStyle(
                    fontSize: 10.5,
                    fontWeight: FontWeight.w800,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: [
              for (final source in sources)
                _SourceChip(raw: source, onOpenLink: onOpenLink),
            ],
          ),
        ],
      ),
    );
  }
}

class _SourceChip extends StatefulWidget {
  const _SourceChip({required this.raw, this.onOpenLink});

  final String raw;
  final LinkLauncher? onOpenLink;

  @override
  State<_SourceChip> createState() => _SourceChipState();
}

class _SourceChipState extends State<_SourceChip> {
  bool _hovering = false;

  void _open(SourceCitation citation) {
    final url = citation.url;
    if (url == null) return;
    unawaited(openExternalLink(url, launcher: widget.onOpenLink));
  }

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    // Format: "[1] informatika_oas_2021.pdf — strana 12"
    //     or: "[2] Osnovi programiranja — https://pmf.kg.ac.rs/..."
    final citation = parseSourceCitation(widget.raw);
    final number = citation.number ?? '·';
    final url = citation.url;
    final clickable = url != null;
    final primary = Theme.of(context).colorScheme.primary;

    return Tooltip(
      message: clickable
          ? '${citation.label}\n${url.toString()}'
          : citation.label,
      child: MouseRegion(
        cursor: clickable ? SystemMouseCursors.click : SystemMouseCursors.basic,
        onEnter: (_) => setState(() => _hovering = true),
        onExit: (_) => setState(() => _hovering = false),
        child: GestureDetector(
          onTap: clickable ? () => _open(citation) : null,
          child: Semantics(
            link: clickable,
            label: clickable
                ? '${citation.label}, otvori izvor'
                : citation.label,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              curve: kEase,
              padding: const EdgeInsets.fromLTRB(6, 5, 12, 5),
              decoration: BoxDecoration(
                color: _hovering
                    ? kIndigo.withValues(alpha: p.isDark ? 0.26 : 0.11)
                    : p.surface,
                borderRadius: BorderRadius.circular(R.pill),
                border: Border.all(
                  color: _hovering ? kIndigo.withValues(alpha: 0.4) : p.border,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 19,
                    height: 19,
                    alignment: Alignment.center,
                    decoration: const BoxDecoration(
                      gradient: kBrandGradient,
                      shape: BoxShape.circle,
                    ),
                    child: Text(
                      number,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 400),
                    child: Text(
                      citation.label,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: clickable ? primary : p.textSecondary,
                        decoration: clickable ? TextDecoration.underline : null,
                        decorationColor: clickable
                            ? primary.withValues(alpha: 0.45)
                            : null,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (clickable) ...[
                    const SizedBox(width: 5),
                    Icon(Icons.open_in_new_rounded, size: 12, color: primary),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------- accents -- //

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;

    return SizedBox(
      height: 20,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, _) {
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (var i = 0; i < 3; i++)
                Padding(
                  padding: EdgeInsets.only(right: i == 2 ? 0 : 5),
                  child: _dot(color, i),
                ),
            ],
          );
        },
      ),
    );
  }

  Widget _dot(Color color, int index) {
    final t = (_controller.value + index * 0.16) % 1.0;
    final wave = math.sin(t * math.pi);

    return Transform.translate(
      offset: Offset(0, -3.5 * wave),
      child: Container(
        width: 7,
        height: 7,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.35 + 0.65 * wave),
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}

class _StreamingCursor extends StatefulWidget {
  const _StreamingCursor();

  @override
  State<_StreamingCursor> createState() => _StreamingCursorState();
}

class _StreamingCursorState extends State<_StreamingCursor>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) => Container(
        width: 8,
        height: 15,
        margin: const EdgeInsets.only(top: 3),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.25 + 0.75 * _controller.value),
          borderRadius: BorderRadius.circular(2),
        ),
      ),
    );
  }
}
