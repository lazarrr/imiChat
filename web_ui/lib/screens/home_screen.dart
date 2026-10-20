import 'package:flutter/material.dart';

import '../state/chat_controller.dart';
import '../theme.dart';
import '../widgets/chat_view.dart';
import '../widgets/sidebar.dart';

/// Responsive shell: permanent sidebar on desktop, drawer on narrow screens.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.controller});

  final ChatController controller;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _scaffoldKey = GlobalKey<ScaffoldState>();

  static const _wideBreakpoint = 900.0;

  void _openSidebar() => _scaffoldKey.currentState?.openDrawer();

  void _closeSidebar() {
    if (_scaffoldKey.currentState?.isDrawerOpen ?? false) {
      Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;

    return Scaffold(
      key: _scaffoldKey,
      body: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth >= _wideBreakpoint) {
            return Row(
              children: [
                Sidebar(controller: controller),
                Expanded(child: ChatView(controller: controller)),
              ],
            );
          }

          return ChatView(
            controller: controller,
            onOpenSidebar: _openSidebar,
          );
        },
      ),
      drawer: MediaQuery.sizeOf(context).width < _wideBreakpoint
          ? _SidebarDrawer(
              controller: controller,
              onSelected: _closeSidebar,
            )
          : null,
    );
  }
}

class _SidebarDrawer extends StatelessWidget {
  const _SidebarDrawer({required this.controller, required this.onSelected});

  final ChatController controller;
  final VoidCallback onSelected;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return Drawer(
      width: 300,
      backgroundColor: p.surface,
      surfaceTintColor: Colors.transparent,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.horizontal(right: Radius.circular(R.lg)),
      ),
      child: SafeArea(
        child: Sidebar(
          controller: controller,
          onConversationSelected: onSelected,
        ),
      ),
    );
  }
}
