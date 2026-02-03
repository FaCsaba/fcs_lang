const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Tokenizer = struct {
    src: []const u8 = &.{},
    byte_idx: usize = 0,

    const Token = struct {
        repr: []const u8,
        kind: Kind,

        const Keywords = std.StaticStringMap(Kind).initComptime(.{
            .{ "legyen", .keyword_legyen },
        });

        const Kind = enum {
            invalid,
            symbol,
            number,
            float,
            string,
            equals,
            lparen,
            rparen,
            newline,
            plus,
            minus,
            star,
            slash,
            keyword_legyen,
        };
    };

    fn nextCodepoint(t: *Tokenizer) u21 {
        var iter = std.unicode.Utf8Iterator{ .bytes = t.src, .i = t.byte_idx };
        defer t.byte_idx = iter.i;
        return iter.nextCodepoint() orelse '\x00';
    }

    fn peekCodepoint(t: Tokenizer) u21 {
        var iter = std.unicode.Utf8Iterator{ .bytes = t.src, .i = t.byte_idx };
        return iter.nextCodepoint() orelse '\x00';
    }

    const State = enum {
        start,
        eat_whitespace,
        number,
        number_dot,
        number_float,
        symbol,
    };

    fn repr(t: *const Tokenizer, start: usize) []const u8 {
        return t.src[start..t.byte_idx];
    }

    fn nextToken(t: *Tokenizer) ?Token {
        var start = t.byte_idx;

        loop: switch (State.start) {
            .start => switch (t.peekCodepoint()) {
                0 => return null,
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number;
                },
                '\n' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .newline };
                },
                ' ', '\t', 11...'\r' => continue :loop .eat_whitespace,
                '(' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .lparen };
                },
                ')' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .rparen };
                },
                '=' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .equals };
                },
                '+' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .plus };
                },
                '-' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .minus };
                },
                '*' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .star };
                },
                '/' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .slash };
                },
                'A'...'Z', 'a'...'z', 128...std.math.maxInt(u21) => {
                    _ = t.nextCodepoint();
                    continue :loop .symbol;
                },
                else => return Token{ .repr = t.repr(start), .kind = .invalid },
            },
            .eat_whitespace => switch (t.peekCodepoint()) {
                '\n' => continue :loop .start,
                ' ', '\t', 11...'\r' => {
                    start += 1;
                    _ = t.nextCodepoint();
                    continue :loop .eat_whitespace;
                },
                else => continue :loop .start,
            },
            .number => switch (t.peekCodepoint()) {
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number;
                },
                '.' => {
                    _ = t.nextCodepoint();
                    continue :loop .number_dot;
                },
                else => return Token{ .repr = t.repr(start), .kind = .number },
            },
            .number_dot => switch (t.peekCodepoint()) {
                '0'...'9' => continue :loop .number_float,
                else => return Token{ .repr = t.repr(start), .kind = .number },
            },
            .number_float => switch (t.peekCodepoint()) {
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number_float;
                },
                else => return Token{ .repr = t.repr(start), .kind = .float },
            },
            .symbol => switch (t.peekCodepoint()) {
                'A'...'Z', 'a'...'z', 128...std.math.maxInt(u21) => {
                    _ = t.nextCodepoint();
                    continue :loop .symbol;
                },
                else => {
                    const symbol = t.repr(start);
                    if (Token.Keywords.has(symbol)) {
                        return Token{ .repr = symbol, .kind = Token.Keywords.get(symbol).? };
                    } else {
                        return Token{ .repr = symbol, .kind = .symbol };
                    }
                },
            },
        }
    }

    fn peekToken(t: *Tokenizer) ?Token {
        const byte_idx = t.byte_idx;
        defer t.byte_idx = byte_idx;
        return t.nextToken();
    }
};

fn expectTokenEqual(repr: []const u8, kind: Tokenizer.Token.Kind, token: ?Tokenizer.Token) !void {
    try std.testing.expectEqualStrings(repr, token.?.repr);
    try std.testing.expectEqual(kind, token.?.kind);
}
test "Tokenizer" {
    var tz1 = Tokenizer{ .src = "legyen kutya = 123" };
    try expectTokenEqual("legyen", .keyword_legyen, tz1.nextToken());
    try expectTokenEqual("kutya", .symbol, tz1.nextToken());
    try expectTokenEqual("=", .equals, tz1.nextToken());
    try expectTokenEqual("123", .number, tz1.nextToken());

    var tz2 = Tokenizer{ .src = "alma\n123.456\nbűnöző" };
    try expectTokenEqual("alma", .symbol, tz2.nextToken());
    try expectTokenEqual("\n", .newline, tz2.nextToken());
    try expectTokenEqual("123.456", .float, tz2.nextToken());
    try expectTokenEqual("\n", .newline, tz2.nextToken());
    try expectTokenEqual("bűnöző", .symbol, tz2.nextToken());
}

// legyen kutya: szöveg = "vakk"
// legyen macska: szöveg = "miáú"
// legyen alma: szöveg | semmi = ""
// kutya = "vakk vakk"
// írd(kutya)

fn newId(T: type) type {
    return enum(T) {
        _,

        pub fn fromInt(self: T) newId(T) {
            return @enumFromInt(self);
        }

        pub fn toInt(self: newId(T)) T {
            return @intFromEnum(self);
        }
    };
}

const Parser = struct {
    tokenizer: Tokenizer,
    alloc: Allocator,
    parse_tree: ParseTree,

    pub const Error = Allocator.Error;

    pub const ParseTree = struct {
        nodes: ArrayList(Node) = .empty,

        const Node = union(enum) {
            stmt: Stmt,
            binop: BinaryOp,
            symbol: []const u8,
            number: i32,
            float: f64,
            // string: []const u8,

            pub const Id = newId(usize);
        };

        const BinaryOp = struct {
            lhs: Node.Id,
            rhs: Node.Id,
            op: Op,

            const Op = enum(u8) {
                assignment,
                addition,
                subtraction,
                multiplication,
                division,

                const OpInfo = struct {
                    precedence: u8,
                    left_assoc: bool,
                };
                fn precedence(op: Op) OpInfo {
                    return switch (op) {
                        .addition, .subtraction => OpInfo{ .precedence = 1, .left_assoc = true },
                        .multiplication, .division => OpInfo{ .precedence = 2, .left_assoc = true },
                        else => unreachable,
                    };
                }

                fn fromToken(kind: Tokenizer.Token.Kind) ?Op {
                    return switch (kind) {
                        .plus => .addition,
                        .minus => .subtraction,
                        .star => .multiplication,
                        .slash => .division,
                        .equals => .assignment,
                        else => null,
                    };
                }
            };
        };

        const Stmt = struct {
            node: Node.Id,
            next: ?Node.Id = null,
        };

        pub fn addNode(self: *ParseTree, alloc: Allocator, node: Node) Error!Node.Id {
            const id = Node.Id.fromInt(self.nodes.items.len);
            try self.nodes.append(alloc, node);
            return id;
        }

        pub fn getNode(self: ParseTree, id: Node.Id) Node {
            return self.nodes.items[id.toInt()];
        }

        pub fn setSibling(self: *ParseTree, node_id: Node.Id, sibling_id: ?Node.Id) void {
            const node = &self.nodes.items[node_id.toInt()];
            node.stmt.next = sibling_id;
        }

        pub fn dump(self: ParseTree, root: Node.Id) void {
            self.dumpRecurse(root, 0);
        }

        fn dumpRecurse(self: ParseTree, node: Node.Id, level: usize) void {
            switch (self.getNode(node)) {
                .stmt => |stmt| {
                    for (0..level) |_| std.debug.print("  ", .{});
                    self.dumpRecurse(stmt.node, level);
                    if (stmt.next) |next| {
                        for (0..level) |_| std.debug.print("  ", .{});
                        self.dumpRecurse(next, level);
                    }
                },
                .binop => |binop| {
                    for (0..level) |_| std.debug.print("  ", .{});
                    std.debug.print("{}:\n", .{binop.op});
                    self.dumpRecurse(binop.lhs, level + 1);
                    self.dumpRecurse(binop.rhs, level + 1);
                },
                .symbol => |symbol| {
                    for (0..level) |_| std.debug.print("  ", .{});
                    std.debug.print("{s}\n", .{symbol});
                },
                .number => |number| {
                    for (0..level) |_| std.debug.print("  ", .{});
                    std.debug.print("{}\n", .{number});
                },
                .float => |float| {
                    for (0..level) |_| std.debug.print("  ", .{});
                    std.debug.print("{}\n", .{float});
                },
                // .string => |string| {
                //     for (0..level) |_| std.debug.print("  ", .{});
                //     std.debug.print("\"{}\"", .{string});
                // },
            }
        }
    };

    pub fn init(alloc: Allocator, src: []const u8) Parser {
        return .{
            .tokenizer = .{ .src = src },
            .alloc = alloc,
            .parse_tree = .{},
        };
    }

    pub fn deinit(parser: *Parser, alloc: Allocator) void {
        parser.parse_tree.nodes.deinit(alloc);
    }

    pub fn parse(self: *Parser) Error!?ParseTree.Node.Id {
        var stmt = try self.parseStmt() orelse return null;
        const root = stmt;
        while (self.tokenizer.peekToken() != null) {
            const nextStmt = try self.parseStmt() orelse break;
            self.parse_tree.setSibling(stmt, nextStmt);
            stmt = nextStmt;
        }
        return root;
    }

    fn parseStmt(self: *Parser) Error!?ParseTree.Node.Id {
        const expr = try self.parseExpr(0) orelse return null;
        const stmt = try self.parse_tree.addNode(self.alloc, .{ .stmt = .{ .node = expr } });
        if (self.tokenizer.peekToken()) |token| {
            switch (token.kind) {
                .newline => _ = self.tokenizer.nextToken(),
                else => std.debug.panic("unexpected token: {}", .{token}),
            }
        }
        return stmt;
    }

    fn parseExpr(self: *Parser, min_prec: u8) Error!?ParseTree.Node.Id {
        var lhs = try self.parsePrimary() orelse return null;

        while (self.tokenizer.peekToken()) |token| {
            if (ParseTree.BinaryOp.Op.fromToken(token.kind)) |op| {
                const prec = op.precedence();
                if (prec.precedence < min_prec) break;

                _ = self.tokenizer.nextToken();

                const next_prec = if (prec.left_assoc) prec.precedence + 1 else prec.precedence;

                const rhs = try self.parseExpr(next_prec) orelse unreachable; // TODO: error report
                lhs = try self.parse_tree.addNode(self.alloc, .{ .binop = .{ .lhs = lhs, .rhs = rhs, .op = op } });
            } else break;
        }

        // TODO: funcall here

        return lhs;
    }

    fn parsePrimary(self: *Parser) Error!?ParseTree.Node.Id {
        const token = self.tokenizer.nextToken() orelse return null;
        // TODO: error report
        return switch (token.kind) {
            .lparen => self.parseExpr(0),
            .number => try self.parse_tree.addNode(self.alloc, .{ .number = std.fmt.parseInt(i32, token.repr, 10) catch unreachable }),
            .float => try self.parse_tree.addNode(self.alloc, .{ .float = std.fmt.parseFloat(f64, token.repr) catch unreachable }),
            .symbol => try self.parse_tree.addNode(self.alloc, .{ .symbol = token.repr }),
            else => std.debug.panic("got {}", .{token}),
        };
    }
};

test "Parser" {
    var parser = Parser.init(std.testing.allocator, "a + b * c + d\n e * 5 * 6\n f / 7 / 8\n");
    defer parser.deinit(std.testing.allocator);
    const root = try parser.parse() orelse unreachable;
    parser.parse_tree.dumpRecurse(root, 0);
}

const Value = packed struct {
    nan_tag: u16,
    payload: u48,

    const NanTag: u16 = @as(u64, @bitCast(std.math.nan(f64))) >> 48;

    // TODO: Change number to int in other places
    const IntTag: u16 = 0x001 | NanTag;

    comptime {
        std.debug.assert(@sizeOf(Value) == 8);
        std.debug.assert(0x7FF8 == NanTag);
    }

    pub fn isFloat(v: Value) bool {
        return v.nan_tag & NanTag != NanTag or v.nan_tag == NanTag;
    }

    pub fn asFloat(v: Value) f64 {
        return @bitCast(v);
    }

    pub fn fromFloat(f: f64) Value {
        return @bitCast(f);
    }

    pub fn isNumber(v: Value) bool {
        return v.nan_tag == IntTag;
    }

    pub fn asNumber(v: Value) i32 {
        const n_s: i48 = @bitCast(v.payload);
        return @truncate(n_s);
    }

    pub fn fromNumber(n: i32) Value {
        const n_s: u32 = @bitCast(n);
        return Value{ .nan_tag = IntTag, .payload = n_s };
    }
};

test "Value" {
    const numbers = [_]i32{ 5, -1, 12, std.math.maxInt(i32), std.math.minInt(i32) };
    for (numbers) |n| {
        const v = Value.fromNumber(n);
        try std.testing.expect(v.isNumber());
        try std.testing.expectEqual(n, v.asNumber());
    }

    const floats = [_]f64{ 0, -1.1, 69, std.math.pi };
    for (floats) |f| {
        const v = Value.fromFloat(f);
        try std.testing.expect(v.isFloat());
        try std.testing.expectEqual(f, v.asFloat());
    }

    const nan = std.math.nan(f64);
    // NaN is special, because it cannot be equal to itself
    try std.testing.expect(Value.fromFloat(nan).isFloat());
}
