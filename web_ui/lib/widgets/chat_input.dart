import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';
import 'controls.dart';

/// The message composer.
///
/// Keyboard contract (matches every chat app users already know):
///   * `Enter` sends the question.
///   * `Shift + Enter` inserts a line break.
///
/// On Flutter web a multiline [TextField] would normally swallow `Enter` and
/// insert a newline, so the shortcut is intercepted here first: a handled key
/// event makes the engine call `preventDefault()` on the DOM keydown, which is
/// what stops the browser from adding the line break.
class ChatInput extends StatefulWidget {
  const ChatInput({
    super.key,
    required this.onSend,
    required this.onStop,
    required this.isStreaming,
    required this.onNewConversation,
    this.enabled = true,
  });

  final ValueChanged<String> onSend;
  final VoidCallback onStop;
  final bool isStreaming;
  final VoidCallback onNewConversation;
  final bool enabled;

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  final _controller = TextEditingController();
  final _focusNode = FocusNode();

  bool _focused = false;
  bool _hasText = false;

  @override
  void initState() {
    super.initState();
    _focusNode.addListener(_handleFocus);
    _controller.addListener(_handleText);
  }

  @override
  void dispose() {
    _focusNode.removeListener(_handleFocus);
    _controller.removeListener(_handleText);
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _handleFocus() {
    if (_focused != _focusNode.hasFocus) {
      setState(() => _focused = _focusNode.hasFocus);
    }
  }

  void _handleText() {
    final hasText = _controller.text.trim().isNotEmpty;
    if (hasText != _hasText) setState(() => _hasText = hasText);
  }

  bool get _canSend =>
      widget.enabled && _hasText && !widget.isStreaming;

  void _submit() {
    if (!_canSend) return;
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    widget.onSend(text);
    _controller.clear();
    _focusNode.requestFocus();
  }

  /// Inserts a line break at the caret.
  ///
  /// Done here instead of leaving it to the browser's default action so that
  /// `Shift + Enter` behaves identically on every browser and platform.
  void _insertLineBreak() {
    final value = _controller.value;
    final text = value.text;

    final selection = value.selection.isValid
        ? value.selection
        : TextSelection.collapsed(offset: text.length);
    final start = selection.start.clamp(0, text.length);
    final end = selection.end.clamp(0, text.length);

    _controller.value = TextEditingValue(
      text: text.replaceRange(start, end, '\n'),
      selection: TextSelection.collapsed(offset: start + 1),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;

    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.enter): _submit,
        const SingleActivator(LogicalKeyboardKey.numpadEnter): _submit,
        const SingleActivator(LogicalKeyboardKey.enter, shift: true):
            _insertLineBreak,
        const SingleActivator(LogicalKeyboardKey.numpadEnter, shift: true):
            _insertLineBreak,
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        curve: kEase,
        decoration: BoxDecoration(
          color: p.surface,
          borderRadius: BorderRadius.circular(R.xl),
          border: Border.all(
            color: _focused
                ? kIndigo.withValues(alpha: p.isDark ? 0.75 : 0.55)
                : p.border,
            width: _focused ? 1.5 : 1,
          ),
          boxShadow: [
            BoxShadow(
              color: p.shadow,
              blurRadius: _focused ? 28 : 18,
              spreadRadius: -6,
              offset: const Offset(0, 10),
            ),
            if (_focused)
              BoxShadow(
                color: kIndigo.withValues(alpha: p.isDark ? 0.26 : 0.16),
                blurRadius: 26,
                spreadRadius: -4,
                offset: const Offset(0, 4),
              ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 8, 14, 4),
              child: TextField(
                controller: _controller,
                focusNode: _focusNode,
                enabled: widget.enabled,
                minLines: 1,
                maxLines: 7,
                textInputAction: TextInputAction.newline,
                keyboardType: TextInputType.multiline,
                // Fallback for IMEs/platforms that report a submit action
                // instead of delivering a raw Enter key event.
                onSubmitted: (_) => _submit(),
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontSize: 14.5,
                  height: 1.5,
                ),
                decoration: InputDecoration(
                  hintText: 'Postavi pitanje o fakultetu…',
                  hintStyle: TextStyle(
                    color: p.textMuted,
                    fontSize: 14.5,
                  ),
                  filled: false,
                  isDense: true,
                  border: InputBorder.none,
                  enabledBorder: InputBorder.none,
                  focusedBorder: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(vertical: 10),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
              child: Row(
                children: [
                  IconAction(
                    icon: Icons.add_comment_outlined,
                    tooltip: 'Novi razgovor',
                    onPressed:
                        widget.isStreaming ? null : widget.onNewConversation,
                  ),
                  const SizedBox(width: 4),
                  Expanded(
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        if (constraints.maxWidth < 240) {
                          return const SizedBox.shrink();
                        }
                        return _ShortcutHint(streaming: widget.isStreaming);
                      },
                    ),
                  ),
                  const SizedBox(width: 8),
                  _SendButton(
                    isStreaming: widget.isStreaming,
                    enabled: _canSend,
                    onSend: _submit,
                    onStop: widget.onStop,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ShortcutHint extends StatelessWidget {
  const _ShortcutHint({required this.streaming});

  final bool streaming;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    if (streaming) {
      return Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 12,
            height: 12,
            child: CircularProgressIndicator(
              strokeWidth: 1.8,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              'Generišem odgovor…',
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 11.5,
                fontWeight: FontWeight.w600,
                color: p.textMuted,
              ),
            ),
          ),
        ],
      );
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _Key(label: 'Enter'),
        const SizedBox(width: 6),
        Flexible(
          child: Text(
            'za slanje',
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              fontSize: 11.5,
              color: p.textMuted,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        const SizedBox(width: 10),
        _Key(label: 'Shift'),
        const SizedBox(width: 3),
        Text(
          '+',
          style: TextStyle(fontSize: 11.5, color: p.textMuted),
        ),
        const SizedBox(width: 3),
        _Key(label: 'Enter'),
        const SizedBox(width: 6),
        Flexible(
          child: Text(
            'novi red',
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              fontSize: 11.5,
              color: p.textMuted,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }
}

class _Key extends StatelessWidget {
  const _Key({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: p.surfaceAlt,
        borderRadius: BorderRadius.circular(R.xs - 2),
        border: Border.all(color: p.border),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10.5,
          fontWeight: FontWeight.w700,
          color: p.textSecondary,
          letterSpacing: 0.1,
        ),
      ),
    );
  }
}

/// Circular send button that morphs into a stop button while streaming.
class _SendButton extends StatefulWidget {
  const _SendButton({
    required this.isStreaming,
    required this.enabled,
    required this.onSend,
    required this.onStop,
  });

  final bool isStreaming;
  final bool enabled;
  final VoidCallback onSend;
  final VoidCallback onStop;

  @override
  State<_SendButton> createState() => _SendButtonState();
}

class _SendButtonState extends State<_SendButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  bool _hovering = false;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1600),
    );
    if (widget.isStreaming) _pulse.repeat();
  }

  @override
  void didUpdateWidget(covariant _SendButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isStreaming && !oldWidget.isStreaming) {
      _pulse.repeat();
    } else if (!widget.isStreaming && oldWidget.isStreaming) {
      _pulse.stop();
      _pulse.value = 0;
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;

    if (widget.isStreaming) {
      return Tooltip(
        message: 'Zaustavi generisanje',
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          onEnter: (_) => setState(() => _hovering = true),
          onExit: (_) => setState(() => _hovering = false),
          child: GestureDetector(
            onTap: widget.onStop,
            child: AnimatedBuilder(
              animation: _pulse,
              builder: (context, child) {
                final t = _pulse.value;
                return Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.errorContainer
                        .withValues(alpha: p.isDark ? 0.32 : 1),
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: kRose.withValues(alpha: 0.35 + 0.45 * t),
                      width: 1.5,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: kRose.withValues(alpha: 0.18 + 0.25 * t),
                        blurRadius: 10 + 12 * t,
                        spreadRadius: -2 + 3 * t,
                      ),
                    ],
                  ),
                  child: Icon(
                    Icons.stop_rounded,
                    size: 19,
                    color: theme.colorScheme.onErrorContainer,
                  ),
                );
              },
            ),
          ),
        ),
      );
    }

    final active = widget.enabled;

    return Tooltip(
      message: 'Pošalji (Enter)',
      child: MouseRegion(
        cursor: active ? SystemMouseCursors.click : SystemMouseCursors.basic,
        onEnter: (_) => setState(() => _hovering = true),
        onExit: (_) => setState(() => _hovering = false),
        child: GestureDetector(
          onTap: active ? widget.onSend : null,
          child: AnimatedScale(
            duration: const Duration(milliseconds: 160),
            curve: kEase,
            scale: active && _hovering ? 1.07 : 1,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              curve: kEase,
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                gradient: active ? kBrandGradient : null,
                color: active ? null : p.surfaceAlt,
                shape: BoxShape.circle,
                border: Border.all(
                  color: active ? Colors.transparent : p.border,
                ),
                boxShadow: active
                    ? [
                        BoxShadow(
                          color: kIndigo.withValues(alpha: 0.42),
                          blurRadius: _hovering ? 18 : 12,
                          spreadRadius: -2,
                          offset: const Offset(0, 4),
                        ),
                      ]
                    : null,
              ),
              child: Icon(
                Icons.arrow_upward_rounded,
                size: 21,
                color: active ? Colors.white : p.textMuted,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
