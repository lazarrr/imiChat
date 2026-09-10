import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../state/chat_controller.dart';
import 'chat_input.dart';
import 'message_bubble.dart';
import 'welcome_view.dart';

class ChatView extends StatefulWidget {
  const ChatView({
    super.key,
    required this.controller,
    required this.onOpenSidebar,
  });

  final ChatController controller;
  final VoidCallback onOpenSidebar;

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
    final nearBottom = _scrollController.position.pixels >=
        _scrollController.position.maxScrollExtent - 120;
    if (nearBottom != !_showJumpButton) {
      setState(() => _showJumpButton = !nearBottom);
    }
  }

  void _scrollToBottom({bool animate = true}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      final target = _scrollController.position.maxScrollExtent;
      if (animate) {
        _scrollController.animateTo(
          target,
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOut,
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
          onOpenSidebar: widget.onOpenSidebar,
        ),
        Expanded(
          child: Stack(
            children: [
              if (messages.isEmpty)
                WelcomeView(onSuggestion: controller.sendMessage)
              else
                ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    final message = messages[index];
                    final shouldScroll =
                        index == messages.length - 1 &&
                            (message.streaming ||
                                message.role == MessageRole.user);
                    if (shouldScroll) _scrollToBottom();
                    return MessageBubble(message: message);
                  },
                ),
              if (_showJumpButton && messages.isNotEmpty)
                Positioned(
                  bottom: 16,
                  right: 20,
                  child: _JumpToBottomButton(onTap: _scrollToBottom),
                ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 20),
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

class _ChatHeader extends StatelessWidget {
  const _ChatHeader({required this.title, required this.onOpenSidebar});

  final String title;
  final VoidCallback onOpenSidebar;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF151824) : Colors.white,
        border: Border(
          bottom: BorderSide(
            color: theme.dividerTheme.color ?? Colors.black12,
          ),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: onOpenSidebar,
            tooltip: 'Meni',
            icon: const Icon(Icons.menu_rounded, size: 21),
          ),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w700,
                letterSpacing: -0.2,
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: isDark
                  ? const Color(0xFF20242F)
                  : const Color(0xFFF0F2F8),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.bolt_rounded,
                    size: 14, color: theme.colorScheme.primary),
                const SizedBox(width: 5),
                Text(
                  'Llama 3.1 8B · RAG',
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _JumpToBottomButton extends StatelessWidget {
  const _JumpToBottomButton({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      elevation: 3,
      color: Theme.of(context).colorScheme.inverseSurface,
      shape: const CircleBorder(),
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Padding(
          padding: const EdgeInsets.all(9),
          child: Icon(
            Icons.keyboard_double_arrow_down_rounded,
            size: 20,
            color: Theme.of(context).colorScheme.onInverseSurface,
          ),
        ),
      ),
    );
  }
}
