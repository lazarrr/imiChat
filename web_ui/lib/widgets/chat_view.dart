import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../state/chat_controller.dart';
import '../theme.dart';
import 'chat_input.dart';
import 'controls.dart';
import 'message_bubble.dart';
import 'welcome_view.dart';

/// Main column: sticky header, message stream, composer.
class ChatView extends StatefulWidget {
  const ChatView({
    super.key,
    required this.controller,
    this.onOpenSidebar,
  });

  final ChatController controller;

  /// When non-null a hamburger button is shown (narrow layouts only).
  final VoidCallback? onOpenSidebar;

  @override
  State<ChatView> createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  final _scrollController = ScrollController();
  bool _showJumpButton = false;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onScroll);
    _scrollController.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (!_scrollController.hasClients) return;
    final position = _scrollController.position;
    final nearBottom = position.pixels >= position.maxScrollExtent - 140;
    final shouldShow = !nearBottom;
    if (shouldShow != _showJumpButton) {
      setState(() => _showJumpButton = shouldShow);
    }
  }

  void _scrollToBottom({bool animate = true}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      final target = _scrollController.position.maxScrollExtent;
      if (animate) {
        _scrollController.animateTo(
          target,
          duration: const Duration(milliseconds: 240),
          curve: kEase,
        );
      } else {
        _scrollController.jumpTo(target);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final messages = controller.currentMessages;

    return Column(
      children: [
        _ChatHeader(
          title: controller.current?.title ?? 'UniChat',
          messageCount: messages.length,
          onOpenSidebar: widget.onOpenSidebar,
        ),
        Expanded(
          child: Stack(
            children: [
              const Positioned.fill(child: _AmbientBackground()),
              if (messages.isEmpty)
                WelcomeView(onSuggestion: controller.sendMessage)
              else
                ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.fromLTRB(20, 22, 20, 26),
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    final message = messages[index];
                    final shouldScroll = index == messages.length - 1 &&
                        (message.streaming ||
                            message.role == MessageRole.user);
                    if (shouldScroll) _scrollToBottom();

                    return Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 940),
                        child: MessageBubble(
                          key: ValueKey(message.id),
                          message: message,
                        ),
                      ),
                    );
                  },
                ),
              if (_showJumpButton && messages.isNotEmpty)
                Positioned(
                  bottom: 18,
                  left: 0,
                  right: 0,
                  child: Center(child: _JumpToBottomButton(onTap: _scrollToBottom)),
                ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 18),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 860),
              child: ChatInput(
                onSend: controller.sendMessage,
                onStop: controller.stopStreaming,
                onNewConversation: controller.newConversation,
                isStreaming: controller.isStreaming,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Two soft colour washes that keep the large empty canvas from looking flat.
class _AmbientBackground extends StatelessWidget {
  const _AmbientBackground();

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return IgnorePointer(
      child: Stack(
        children: [
          Positioned(
            top: -160,
            left: -120,
            child: _Blob(color: p.glowA, size: 460),
          ),
          Positioned(
            bottom: -180,
            right: -120,
            child: _Blob(color: p.glowB, size: 520),
          ),
        ],
      ),
    );
  }
}

class _Blob extends StatelessWidget {
  const _Blob({required this.color, required this.size});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [color, color.withValues(alpha: 0)],
        ),
      ),
    );
  }
}

class _ChatHeader extends StatelessWidget {
  const _ChatHeader({
    required this.title,
    required this.messageCount,
    required this.onOpenSidebar,
  });

  final String title;
  final int messageCount;
  final VoidCallback? onOpenSidebar;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;

    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 16, 10),
      decoration: BoxDecoration(
        color: p.surface,
        border: Border(bottom: BorderSide(color: p.border)),
      ),
      child: Row(
        children: [
          if (onOpenSidebar != null)
            IconAction(
              icon: Icons.menu_rounded,
              tooltip: 'Meni',
              onPressed: onOpenSidebar,
            ),
          if (onOpenSidebar != null) const SizedBox(width: 4),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.25,
                  ),
                ),
                const SizedBox(height: 1),
                Text(
                  messageCount == 0
                      ? 'Nova konverzacija'
                      : '$messageCount ${_plural(messageCount)}',
                  style: TextStyle(
                    fontSize: 11.5,
                    color: p.textMuted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          LayoutBuilder(
            builder: (context, constraints) {
              if (constraints.maxWidth < 300) return const SizedBox.shrink();
              return const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Pill(
                    label: 'Llama 3.1 8B',
                    icon: Icons.bolt_rounded,
                    dense: true,
                  ),
                  SizedBox(width: 7),
                  Pill(
                    label: 'RAG',
                    icon: Icons.hub_rounded,
                    dense: true,
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

String _plural(int count) {
  final mod10 = count % 10;
  final mod100 = count % 100;
  if (mod10 == 1 && mod100 != 11) return 'poruka';
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return 'poruke';
  }
  return 'poruka';
}

class _JumpToBottomButton extends StatefulWidget {
  const _JumpToBottomButton({required this.onTap});

  final VoidCallback onTap;

  @override
  State<_JumpToBottomButton> createState() => _JumpToBottomButtonState();
}

class _JumpToBottomButtonState extends State<_JumpToBottomButton> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          curve: kEase,
          padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 9),
          decoration: BoxDecoration(
            color: p.surface,
            borderRadius: BorderRadius.circular(R.pill),
            border: Border.all(
              color: _hovering ? kIndigo.withValues(alpha: 0.45) : p.border,
            ),
            boxShadow: [
              BoxShadow(
                color: p.shadow,
                blurRadius: _hovering ? 20 : 14,
                spreadRadius: -4,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.arrow_downward_rounded,
                size: 15,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(width: 7),
              Text(
                'Na dno',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: p.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
