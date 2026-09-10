import 'package:flutter/material.dart';

import '../state/chat_controller.dart';
import '../widgets/chat_view.dart';
import '../widgets/sidebar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.controller});

  final ChatController controller;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _scaffoldKey = GlobalKey<ScaffoldState>();

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
          final wide = constraints.maxWidth >= 900;

          if (wide) {
            return Row(
              children: [
                Sidebar(controller: controller),
                Expanded(
                  child: ChatView(
                    controller: controller,
                    onOpenSidebar: () {},
                  ),
                ),
              ],
            );
          }

          return ChatView(
            controller: controller,
            onOpenSidebar: _openSidebar,
          );
        },
      ),
      drawer: MediaQuery.sizeOf(context).width < 900
          ? Drawer(
              width: 300,
              child: SafeArea(
                child: Sidebar(
                  controller: controller,
                  onConversationSelected: _closeSidebar,
                ),
              ),
            )
          : null,
    );
  }
}
